from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import time
from pydantic import BaseModel, Field
from typing import List, Optional
from src.routes import router


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
            app.state.artifacts["model"] = model
        except Exception as e:
            print(f"Error loading model: {e}")
            
    app.state.artifacts["movies"] = load_movies_metadata()
    yield
    app.state.artifacts.clear()


app = FastAPI(
    title="Movie Lens 2-Stage Recommender API",
    lifespan=lifespan
)

app.state.artifacts = {}

app.include_router(router)
