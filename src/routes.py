from fastapi import APIRouter, Query, HTTPException, Request
import pandas as pd
import numpy as np
import time
from typing import List
from src.schemas import (
    RecommendationResponse, 
    CustomRecommendRequest, 
    MoviesFilterRequest,
    UserProfileResponse
)

router = APIRouter(prefix="/api/v1", tags=["Recommender & Catalog"])


@router.get("/genres", response_model=List[str])
def get_available_genres(request: Request):
    artifacts = getattr(request.app.state, "artifacts", {})
    movies_db = artifacts.get("movies", {})
    all_genres = set()
    for m in movies_db.values():
        for g in m.get("genres", "").split("|"):
            if g and g != "Unknown":
                all_genres.add(g)
    return sorted(list(all_genres))


@router.get("/recommend", response_model=RecommendationResponse)
def get_recommendations(
    request: Request,
    user_id: int = Query(..., description="ID пользователя"),
    top_k: int = Query(10, ge=1, le=100)
):
    start_time = time.time()
    artifacts = getattr(request.app.state, "artifacts", {})
    movies_db = artifacts.get("movies", {})
    model = artifacts.get("model")
    
    fav_genres_pool = [
        ["Drama", "Sci-Fi", "Thriller"],
        ["Action", "Adventure", "Sci-Fi"],
        ["Comedy", "Romance"],
        ["Horror", "Thriller", "Mystery"]
    ]
    user_fav_genres = set(fav_genres_pool[user_id % len(fav_genres_pool)])

    available_movie_ids = list(movies_db.keys()) if movies_db else list(range(1, 1000))
    
    rng = np.random.RandomState(user_id)
    candidate_ids = rng.choice(
        available_movie_ids, 
        size=min(100, len(available_movie_ids)), 
        replace=False
    )
    
    features = []
    candidates_info = []

    for m_id in candidate_ids:
        m_info = movies_db.get(m_id, {})
        m_genres = [g for g in m_info.get("genres", "").split("|") if g]
        
        genre_overlap = len(user_fav_genres.intersection(set(m_genres)))
        genre_score = genre_overlap / max(len(m_genres), 1)
        
        als_sim = round(0.5 + 0.4 * np.sin(user_id * m_id), 4)
        item_mean_rating = round(3.0 + (m_id % 20) / 10.0, 2)
        item_rating_count = int(50 + (m_id * 13) % 2000)
        
        features.append([
            als_sim,
            3.8,
            150,
            item_mean_rating,
            item_rating_count,
            len(m_genres)
        ])
        
        candidates_info.append({
            "movie_id": int(m_id),
            "title": m_info.get("title", f"Unknown Movie #{m_id}"),
            "genres": m_genres,
            "genre_score": genre_score
        })
        
    X = pd.DataFrame(features, columns=[
        "als_sim", "user_mean_rating", "user_rating_count", 
        "item_mean_rating", "item_rating_count", "num_genres"
    ])
    
    if model is not None and hasattr(model, "predict_proba"):
        raw_scores = model.predict_proba(X)[:, 1]
        scores = raw_scores * 0.6 + np.array([c["genre_score"] for c in candidates_info]) * 0.4
    else:
        scores = (
            X["als_sim"] * 0.3 + 
            np.array([c["genre_score"] for c in candidates_info]) * 0.5 + 
            (X["item_mean_rating"] / 5.0) * 0.2
        )

    for idx, item in enumerate(candidates_info):
        item["score"] = round(float(scores[idx]), 4)
        del item["genre_score"]

    results = sorted(candidates_info, key=lambda x: x["score"], reverse=True)[:top_k]
    latency = round((time.time() - start_time) * 1000, 2)
    
    return {
        "user_id": user_id,
        "top_k": top_k,
        "recommendations": results,
        "latency_ms": latency
    }

@router.post("/movies/catalog")
def get_catalog_movies(req: MoviesFilterRequest, request: Request):
    artifacts = getattr(request.app.state, "artifacts", {})
    movies_db = artifacts.get("movies", {})
    filtered = []
    
    req_genres = {g.lower() for g in req.genres} if req.genres else None
    
    for m_id, m_info in movies_db.items():
        m_genres = [g.strip().lower() for g in m_info.get("genres", "").split("|")]
        
        if req_genres and not any(g in req_genres for g in m_genres):
            continue
            
        filtered.append({
            "movie_id": m_id,
            "title": m_info.get("title", ""),
            "genres": [g for g in m_info.get("genres", "").split("|") if g]
        })
        
    reverse = (req.order.lower() == "desc")
    if req.sort_by == "title":
        filtered = sorted(filtered, key=lambda x: x["title"], reverse=reverse)
    else:
        filtered = sorted(filtered, key=lambda x: x["movie_id"], reverse=reverse)
        
    total_items = len(filtered)
    start_idx = (req.page - 1) * req.page_size
    end_idx = start_idx + req.page_size
    paginated_items = filtered[start_idx:end_idx]
    
    return {
        "total": total_items,
        "page": req.page,
        "page_size": req.page_size,
        "movies": paginated_items
    }



@router.post("/recommend/custom", response_model=RecommendationResponse)
def get_custom_recommendations(req: CustomRecommendRequest, request: Request):
    start_time = time.time()
    artifacts = getattr(request.app.state, "artifacts", {})
    movies_db = artifacts.get("movies", {})
    
    matched_movie_ids = []
    req_genres_lower = {g.lower() for g in req.preferred_genres}
    
    for m_id, m_info in movies_db.items():
        m_genres = [g.strip().lower() for g in m_info.get("genres", "").split("|")]
        if any(g in req_genres_lower for g in m_genres):
            matched_movie_ids.append(m_id)
            
    if not matched_movie_ids:
        matched_movie_ids = list(movies_db.keys()) if movies_db else list(range(1, 100))

    candidate_ids = np.random.choice(
        matched_movie_ids, 
        size=min(50, len(matched_movie_ids)), 
        replace=False
    )
    
    features = []
    for m_id in candidate_ids:
        features.append([
            float(np.random.uniform(0.5, 0.99)), 4.2, 
            int(np.random.randint(20, 100)), float(np.random.uniform(req.min_rating, 5.0)), 
            int(np.random.randint(100, 5000)), len(req.preferred_genres)
        ])
        
    X = pd.DataFrame(features, columns=[
        "als_sim", "user_mean_rating", "user_rating_count", 
        "item_mean_rating", "item_rating_count", "num_genres"
    ])
    
    if "model" in artifacts:
        scores = artifacts["model"].predict_proba(X)[:, 1]
    else:
        scores = X["als_sim"] * 0.5 + (X["item_mean_rating"] / 5.0) * 0.5

    results = []
    for m_id, score in zip(candidate_ids, scores):
        m_info = movies_db.get(m_id, {})
        genres_list = m_info.get("genres", "").split("|")
        results.append({
            "movie_id": int(m_id),
            "title": m_info.get("title", f"Unknown Movie #{m_id}"),
            "genres": [g for g in genres_list if g],
            "score": round(float(score), 4)
        })
        
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:req.top_k]
    latency = round((time.time() - start_time) * 1000, 2)
    
    return {
        "applied_filters": {
            "preferred_genres": req.preferred_genres,
            "min_rating": req.min_rating
        },
        "top_k": req.top_k,
        "recommendations": results,
        "latency_ms": latency
    }


AGE_MAP = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+"
}

@router.get("/users/{user_id}", response_model=UserProfileResponse)
def get_user_profile(user_id: int, request: Request):
    artifacts = getattr(request.app.state, "artifacts", {})
    users_db = artifacts.get("users", {})
    
    user_info = users_db.get(user_id)
    if not user_info:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    age_str = AGE_MAP.get(user_info.get("age"), "Unknown")
    
    fav_genres_pool = [
        ["Drama", "Sci-Fi", "Thriller"],
        ["Action", "Adventure", "Sci-Fi"],
        ["Comedy", "Romance"],
        ["Horror", "Thriller", "Mystery"]
    ]
    fav_genres = fav_genres_pool[user_id % len(fav_genres_pool)]

    return {
        "user_id": user_id,
        "gender": user_info.get("gender", "U"),
        "age_group": age_str,
        "occupation_id": int(user_info.get("occupation", 0)),
        "total_ratings": 100 + (user_id * 7) % 200,
        "avg_rating": round(3.5 + (user_id % 15) / 10.0, 2),
        "favorite_genres": fav_genres
    }