from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    score: float

class RecommendationResponse(BaseModel):
    user_id: Optional[int] = None
    applied_filters: Optional[dict] = None
    top_k: int
    recommendations: List[RecommendationItem]
    latency_ms: float

class CustomRecommendRequest(BaseModel):
    preferred_genres: List[str] = Field(
        default=["Action", "Sci-Fi"], 
        description="Список любимых жанров"
    )
    min_rating: Optional[float] = Field(
        default=3.5, ge=1.0, le=5.0, 
        description="Минимальный рейтинг"
    )
    top_k: Optional[int] = Field(default=10, ge=1, le=50)

class MoviesFilterRequest(BaseModel):
    genres: Optional[List[str]] = Field(default=None, description="Фильтр по жанрам")
    min_rating: Optional[float] = Field(default=0.0, ge=0.0, le=5.0)
    sort_by: str = Field(default="rating", description="Сортировка: rating или title")
    order: str = Field(default="desc", description="Порядок: asc или desc")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=100)


