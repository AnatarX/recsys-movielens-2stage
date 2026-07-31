import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import implicit
import faiss


class ALSRetriever:
    def __init__(self, factors: int = 64, iterations: int = 15, regularization: float = 0.05):
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        
        self.model = None
        self.index = None
        
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}

    def _create_mappings(self, df: pd.DataFrame):
        unique_users = df["user_id"].unique()
        unique_items = df["movie_id"].unique()

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = {i: u for i, u in enumerate(unique_users)}
        
        self.item2idx = {item: i for i, item in enumerate(unique_items)}
        self.idx2item = {i: item for i, item in enumerate(unique_items)}

    def fit(self, train_df: pd.DataFrame):
        print("\nОбучаем ALS модель (Stage 1 Retrieval)...")
        self._create_mappings(train_df)

        rows = train_df["user_id"].map(self.user2idx).values
        cols = train_df["movie_id"].map(self.item2idx).values
        data = np.ones(len(train_df), dtype=np.float32)

        user_item_matrix = csr_matrix(
            (data, (rows, cols)), 
            shape=(len(self.user2idx), len(self.item2idx))
        )

        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=42
        )
        self.model.fit(user_item_matrix)

        print("Indexing embbedings фильмов в Faiss...")
        item_vectors = self.model.item_factors
        
        faiss.normalize_L2(item_vectors)

        self.index = faiss.IndexFlatIP(self.factors)
        self.index.add(item_vectors)
        print(f"Indexed {self.index.ntotal} movies")

    def generate_candidates(self, user_ids: list[int], top_k: int = 100) -> dict[int, list[int]]:
        candidates = {}

        valid_user_indices = []
        target_user_ids = []

        for uid in user_ids:
            if uid in self.user2idx:
                valid_user_indices.append(self.user2idx[uid])
                target_user_ids.append(uid)

        if not valid_user_indices:
            return candidates

        user_vectors = self.model.user_factors[valid_user_indices]
        faiss.normalize_L2(user_vectors)

        _, nearest_indices = self.index.search(user_vectors, top_k)

        for u_id, item_idxs in zip(target_user_ids, nearest_indices):
            candidates[u_id] = [self.idx2item[idx] for idx in item_idxs if idx in self.idx2item]

        return candidates