import os

import urllib
import zipfile

import pandas as pd
import urllib.request


class MovieLensDataLoader:
    URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")

        os.makedirs(self.raw_dir, exist_ok = True)

    def download_and_extract(self):
        zip_path = os.path.join(self.raw_dir, "ml-1m.zip")
        extracted_path = os.path.join(self.raw_dir, "ml-1m")

        if not os.path.exists(extracted_path):
            print("[WORKER] MovieLens 1M downloading...")
            urllib.request.urlretrieve(self.URL, zip_path)

            print("[WORKER] Unpacking...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(self.raw_dir)
            print("[WORKER] Succesfuly downloaded.")
        else:
            print("[WORKER] Already downloaded.")

    def load_raw_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.download_and_extract()
        base_path = os.path.join(self.raw_dir, "ml-1m")

        ratings = pd.read_csv(
            os.path.join(base_path, "ratings.dat"),
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )

        movies = pd.read_csv(
            os.path.join(base_path, "movies.dat"),
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
        )

        users = pd.read_csv(
            os.path.join(base_path, "users.dat"),
            sep="::",
            engine="python",
            names = ["user_id", "gender", "age", "occupation", "zip_code"]
        )

        return ratings, movies, users

    @staticmethod
    def time_based_split(ratings: pd.DataFrame, test_size_ratio: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
        print("[WORKER] Time-based split...")

        ratings = ratings.sort_values(["user_id", "timestamp"]).reset_index(
            drop=True
        )

        ratings["rank"] = ratings.groupby("user_id")["timestamp"].rank(
            method="first", ascending=True
        )

        ratings["user_total"] = ratings.groupby("user_id")["user_id"].transform(
            "count"
        )

        test_condition = ratings["rank"] > (
            ratings["user_total"] * (1 - test_size_ratio)
        )

        train = ratings[~test_condition].drop(
            columns=["rank", "user_total"]
        )

        test = ratings[test_condition].drop(columns=["rank", "user_total"])

        print(f"Train interactions: {len(train):,}")
        print(f"Test interactions: {len(test):,}")

        return train, test