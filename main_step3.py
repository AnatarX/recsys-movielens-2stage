from catboost import CatBoostClassifier
from src.data_loader import MovieLensDataLoader
from src.retrieval import ALSRetriever
from src.features import FeatureExtractor
from src.metrics import evaluate_recommendations


def main():
    loader = MovieLensDataLoader()
    ratings, movies, users = loader.load_raw_data()
    positive_ratings = ratings[ratings["rating"] >= 4.0]

    train, test = loader.time_based_split(positive_ratings, test_size_ratio=0.2)
    test_gt = test.groupby("user_id")["movie_id"].apply(list).to_dict()

    retriever = ALSRetriever(factors=64, iterations=15)
    retriever.fit(train)

    train_users = train["user_id"].unique()
    train_candidates = retriever.generate_candidates(train_users, top_k=100)
    
    test_users = list(test_gt.keys())
    test_candidates = retriever.generate_candidates(test_users, top_k=100)

    fe = FeatureExtractor()
    fe.fit(train, movies)

    train_rerank_df = fe.build_rerank_dataset(train_candidates, train, retriever, is_train=True)
    test_rerank_df = fe.build_rerank_dataset(test_candidates, test, retriever, is_train=False)

    feature_cols = [
        "als_sim", 
        "user_mean_rating", 
        "user_rating_count", 
        "item_mean_rating", 
        "item_rating_count", 
        "num_genres"
    ]

    X_train, y_train = train_rerank_df[feature_cols], train_rerank_df["target"]
    X_test, y_test = test_rerank_df[feature_cols], test_rerank_df["target"]

    print("\nCatBoost Reranker...")
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.08,
        depth=6,
        eval_metric="Logloss",
        random_seed=42,
        verbose=50
    )
    model.fit(X_train, y_train, eval_set=(X_test, y_test))

    test_rerank_df["score"] = model.predict_proba(X_test)[:, 1]

    catboost_preds = {}
    for user_id, group in test_rerank_df.groupby("user_id"):
        top_items = group.sort_values("score", ascending=False)["movie_id"].head(10).tolist()
        catboost_preds[user_id] = top_items

    metrics = evaluate_recommendations(test_gt, catboost_preds, k=10)
    print("\n FINAL RECSYS METRICS (ALS + Faiss + CatBoost):")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

    print("\n Feature Importances:")
    importances = model.get_feature_importance()
    for col, imp in zip(feature_cols, importances):
        print(f"  {col}: {imp:.2f}%")

    model.save_model("catboost_reranker.cbm")

if __name__ == "__main__":
    main()