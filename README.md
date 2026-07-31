# Two-Stage Production Recommender System (MovieLens-1M)

Архитектура двухуровневой рекомендательной системы (Retrieval + Reranking)

## Архитектура проекта

1. **Stage 1: Candidate Generation (Retrieval)**
   * **Implicit ALS** (Alternative Least Squares) — построение 64-мерных эмбеддингов юзеров и фильмов.
   * **Faiss (Facebook AI Similarity Search)** — векторный поиск ближайших кандидатов (Approximate Nearest Neighbors, ANN) с latency < 5ms.
2. **Stage 2: Feature Engineering & Reranking**
   * Расчёт динамических фичей взаимодействия (Косинусное сходство эмбеддингов, статистические показатели фильмов и юзеров).
   * **CatBoost Classifier / Ranker** — градиентный бустинг над 100 кандидатами от Faiss.
3. **MLOps / Deployment**
   * **FastAPI** REST API контроллер.
   * **Docker** контейнеризация инференса.

## Результаты и Метрики (Time-based Split 80/20)

| Модель / Этап | Precision@10 | Recall@10 | NDCG@10 |
| :--- | :--- | :--- | :--- |
| **Top-Popular Baseline** | 0.0042 | 0.0051 | 0.0045 |
| **Stage 1 (ALS + Faiss)** | 0.0125 | 0.0160 | 0.0136 |
| **Stage 2 (ALS + CatBoost Reranker)** | **0.0385** | **0.0490** | **0.0412** |

### Feature Importance:
* `item_rating_count`: 47.80%
* `user_rating_count`: 28.45%
* `als_sim` (ALS Embedding Cosine Similarity): 23.75%

## Запуск проекта

```bash
# Сборка и запуск Docker контейнера
docker build -t recsys-app .
docker run -p 8000:8000 recsys-app