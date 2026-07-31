import pandas as pd
import numpy as np
from tqdm import tqdm


class FeatureExtractor:
    def __init__(self):
        self.user_stats = None
        self.item_stats = None
        self.user_favorite_genres = {}

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame):
        
        self.item_stats = train_df.groupby("movie_id").agg(
            item_mean_rating=("rating", "mean"),
            item_rating_count=("rating", "count")
        ).reset_index()

        movies_df = movies_df.copy()
        movies_df["genre_list"] = movies_df["genres"].str.split("|")
        movies_df["num_genres"] = movies_df["genre_list"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        
        self.item_stats = self.item_stats.merge(
            movies_df[["movie_id", "num_genres", "genre_list"]], on="movie_id", how="left"
        )

        self.user_stats = train_df.groupby("user_id").agg(
            user_mean_rating=("rating", "mean"),
            user_rating_count=("rating", "count")
        ).reset_index()

        train_with_genres = train_df.merge(movies_df[["movie_id", "genre_list"]], on="movie_id", how="left")
        
        user_genres = train_with_genres.explode("genre_list")
        fav_genres = user_genres.groupby(["user_id", "genre_list"]).size().unstack(fill_value=0)
        
        fav_genres_ratio = fav_genres.div(fav_genres.sum(axis=1), axis=0)
        self.user_favorite_genres = fav_genres_ratio.to_dict(orient="index")

    def build_rerank_dataset(
        self, 
        candidates_dict: dict[int, list[int]], 
        target_df: pd.DataFrame, 
        retriever,
        is_train: bool = True
    ) -> pd.DataFrame:
        
        rows = []
        target_set = set(zip(target_df["user_id"], target_df["movie_id"]))

        user_factors = retriever.model.user_factors
        item_factors = retriever.model.item_factors

        for user_id, items in tqdm(candidates_dict.items()):
            if user_id not in retriever.user2idx:
                continue
                
            u_idx = retriever.user2idx[user_id]
            u_vec = user_factors[u_idx]
            u_favs = self.user_favorite_genres.get(user_id, {})

            for item_id in items:
                if item_id not in retriever.item2idx:
                    continue
                    
                i_idx = retriever.item2idx[item_id]
                i_vec = item_factors[i_idx]

                sim = np.dot(u_vec, i_vec) / (np.linalg.norm(u_vec) * np.linalg.norm(i_vec) + 1e-9)
                
                target = 1 if (user_id, item_id) in target_set else 0

                rows.append({
                    "user_id": user_id,
                    "movie_id": item_id,
                    "als_sim": float(sim),
                    "target": target
                })

        df = pd.DataFrame(rows)

        df = df.merge(self.user_stats, on="user_id", how="left")
        df = df.merge(self.item_stats.drop(columns=["genre_list"]), on="movie_id", how="left")

        df.fillna(0, inplace=True)
        return df