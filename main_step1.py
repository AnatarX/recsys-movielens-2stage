from src.data_loader import MovieLensDataLoader
from src.metrics import evaluate_recommendations


def main():
    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_raw_data()

    print(f"\n Всего оценок: {len(ratings):,}")
    print(f" Пользователей: {ratings['user_id'].nunique():,}")
    print(f" Фильмов: {ratings['movie_id'].nunique():,}")

    positive_ratings = ratings[ratings["rating"] >= 4.0]

    train, test = loader.time_based_split(positive_ratings, test_size_ratio=0.2)

    test_gt = test.groupby("user_id")["movie_id"].apply(list).to_dict()

    top_popular_items = (
        train.groupby("movie_id")["rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    popular_preds = {user_id: top_popular_items for user_id in test_gt.keys()}

    metrics = evaluate_recommendations(test_gt, popular_preds, k=10)
    print("\n Top-Popular Baseline Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()