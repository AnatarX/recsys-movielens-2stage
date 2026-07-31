import numpy as np
import pandas as pd


def precision_at_k(actual: list, predicted: list, k: int = 10) -> float:
    predicted_at_k = predicted[:k]
    relevant_count = len(set(predicted_at_k) & set(actual))
    return relevant_count / k


def recall_at_k(actual: list, predicted: list, k: int = 10) -> float:
    if not actual:
        return 0.0
    predicted_at_k = predicted[:k]
    relevant_count = len(set(predicted_at_k) & set(actual))
    return relevant_count / len(actual)


def ndcg_at_k(actual: list, predicted: list, k: int = 10) -> float:
    predicted_at_k = predicted[:k]
    dcg = 0.0
    for i, p in enumerate(predicted_at_k):
        if p in actual:
            dcg += 1.0 / np.log2(i + 2)

    idcg = sum(
        [1.0 / np.log2(i + 2) for i in range(min(len(actual), k))]
    )
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(
    ground_truth: dict[int, list[int]],
    predictions: dict[int, list[int]],
    k: int = 10,
) -> dict[str, float]:
    precisions, recalls, ndcgs = [], [], []

    for user_id, target_items in ground_truth.items():
        if user_id in predictions:
            pred_items = predictions[user_id]
            precisions.append(precision_at_k(target_items, pred_items, k))
            recalls.append(recall_at_k(target_items, pred_items, k))
            ndcgs.append(ndcg_at_k(target_items, pred_items, k))

    return {
        f"Precision@{k}": float(np.mean(precisions)),
        f"Recall@{k}": float(np.mean(recalls)),
        f"NDCG@{k}": float(np.mean(ndcgs)),
    }