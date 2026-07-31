from src.data_loader import MovieLensDataLoader
from src.metrics import evaluate_recommendations
from src.retrieval import ALSRetriever


def main():
    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_raw_data()

    positive_ratings = ratings[ratings["rating"] >= 4.0]

    train, test = loader.time_based_split(positive_ratings, test_size_ratio=0.2)
    test_gt = test.groupby("user_id")["movie_id"].apply(list).to_dict()

    retriever = ALSRetriever(factors=64, iterations=15)
    retriever.fit(train)

    user_ids = list(test_gt.keys())
    predictions_10 = retriever.generate_candidates(user_ids, top_k=10)

    metrics_10 = evaluate_recommendations(test_gt, predictions_10, k=10)
    print("\n ALS + Faiss Metrics (Top-10):")
    for metric_name, value in metrics_10.items():
        print(f"  {metric_name}: {value:.4f}")

    predictions_100 = retriever.generate_candidates(user_ids, top_k=100)
    metrics_100 = evaluate_recommendations(test_gt, predictions_100, k=100)
    print(f"\n Candidate Retrieval Capacity (Recall@100): {metrics_100['Recall@100']:.4f}")


if __name__ == "__main__":
    main()