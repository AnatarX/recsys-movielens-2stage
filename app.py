from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import time

artifacts = {}


def load_movies_metadata():
    movies_path = "data/raw/ml-1m/movies.dat"
    if os.path.exists(movies_path):
        movies = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1"
        )
        return movies.set_index("movie_id").to_dict(orient="index")
    return {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = "catboost_reranker.cbm"
    if os.path.exists(model_path):
        try:
            model = CatBoostClassifier()
            model.load_model(model_path)
            artifacts["model"] = model
        except Exception as e:
            print(f"Error loading model: {e}")
            
    artifacts["movies"] = load_movies_metadata()
    yield
    artifacts.clear()

app = FastAPI(
    title="Movie Lens 2-Stage Recommender API",
    lifespan=lifespan
)

@app.get("/api/v1/recommend")
def get_recommendations(
    user_id: int = Query(..., description="ID пользователя"),
    top_k: int = Query(10, ge=1, le=100)
):
    start_time = time.time()
    
    if "model" not in artifacts:
        raise HTTPException(status_code=500, detail="No model loaded")

    model = artifacts["model"]

    np.random.seed(user_id)

    candidate_ids = np.random.choice(range(1, 1000), size=50, replace=False)
    
    features = []
    for m_id in candidate_ids:
        als_sim = float(np.random.uniform(0.1, 0.99))
        user_mean = 4.1
        user_count = int(np.random.randint(10, 200))
        item_mean = float(np.random.uniform(3.0, 4.8))
        item_count = int(np.random.randint(50, 5000))
        num_genres = int(np.random.randint(1, 5))
        
        features.append([als_sim, user_mean, user_count, item_mean, item_count, num_genres])
        
    X = pd.DataFrame(features, columns=[
        "als_sim", "user_mean_rating", "user_rating_count", 
        "item_mean_rating", "item_rating_count", "num_genres"
    ])
    
    scores = model.predict_proba(X)[:, 1]
    
    movies_db = artifacts.get("movies", {})
    results = []
    
    for m_id, score in zip(candidate_ids, scores):
        m_info = movies_db.get(m_id, {})
        raw_genres = m_info.get("genres", "Unknown")
        genres_list = raw_genres.split("|") if raw_genres != "Unknown" else []

        results.append({
            "movie_id": int(m_id),
            "title": m_info.get("title", f"Unknown Movie #{m_id}"),
            "genres": genres_list,
            "score": round(float(score), 4)
        })
        
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]
    
    latency = round((time.time() - start_time) * 1000, 2)
    
    return {
        "user_id": user_id,
        "top_k": top_k,
        "recommendations": results,
        "latency_ms": latency
    }