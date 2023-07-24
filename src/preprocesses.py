from functools import partial
from typing import Any, Callable, TypedDict

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class TrainedModels(TypedDict):
    watch_histories: Word2Vec | None
    user2producers: Word2Vec | None
    source_le: LabelEncoder | None
    rating_encoder: OneHotEncoder | None


def cast_episode(df: pd.DataFrame) -> pd.DataFrame:
    df["episodes"] = df["episodes"].apply(lambda x: int(x) if x != "Unknown" else -999)
    return df


def user_anime_histories_vec(
    df: pd.DataFrame, trained_model_callback: TrainedModels, user2anime: pd.DataFrame
) -> pd.DataFrame:
    """ユーザーの視聴履歴をベクトル化する"""

    # train+testのuserとanimeの組み合わせのdf
    watch_histories = (
        user2anime.loc[:, ["user_id", "anime_id"]]
        .groupby("user_id")
        .apply(lambda x: x["anime_id"].tolist())
    )
    watch_histories_model = Word2Vec(
        watch_histories, vector_size=10, min_count=1, window=20
    )
    trained_model_callback["watch_histories"] = watch_histories_model

    def get_mean_anime_vec(anime_ids: list[int]) -> np.ndarray:
        watch_vecs = list(map(lambda x: watch_histories_model.wv[str(x)], anime_ids))
        return np.mean(watch_vecs, axis=0)

    user_features = dict(
        map(lambda x: (x[0], get_mean_anime_vec(x[1])), watch_histories.items())
    )
    user_features_df = pd.DataFrame.from_dict(user_features, orient="index")
    user_features_df.columns = [f"watch_histories_vec_{i}" for i in range(10)]
    df = pd.merge(df, user_features_df, left_on="user_id", right_index=True)
    return df


def make_user_anime_histories(
    df: pd.DataFrame, trained_models: TrainedModels, user2anime: pd.DataFrame
) -> pd.DataFrame:
    """ユーザーの視聴履歴をベクトル化する"""

    # userとanimeの組み合わせのdf
    watch_histories = (
        user2anime.loc[:, ["user_id", "anime_id"]]
        .groupby("user_id")
        .apply(lambda x: x["anime_id"].tolist())
    )
    watch_histories_model = trained_models["watch_histories"]
    if watch_histories_model is None:
        raise ValueError("watch_histories_model is None")

    # これすると埋め込みが変わってしまう
    # watch_histories_model.build_vocab(watch_histories, update=True)
    # watch_histories_model.train(
    #     watch_histories,
    #     total_examples=watch_histories_model.corpus_count,
    #     epochs=watch_histories_model.epochs,
    # )

    def get_mean_anime_vec(user_items: tuple[str, list[str]]) -> np.ndarray:
        def _get_vec(x: str) -> np.ndarray:
            try:
                return watch_histories_model.wv[str(x)]
            except KeyError:
                return np.array([-1.0] * watch_histories_model.vector_size)

        watch_vecs = list(map(_get_vec, user_items[1]))
        return np.mean(watch_vecs, axis=0)

    # 予測対象のユーザーの視聴履歴をベクトル化してユーザを特徴づける
    user_features = dict(
        map(lambda x: (x[0], get_mean_anime_vec(x)), watch_histories.items())
    )
    user_features_df = pd.DataFrame.from_dict(user_features, orient="index")
    user_features_df.columns = [f"watch_histories_vec_{i}" for i in range(10)]
    df = pd.merge(df, user_features_df, left_on="user_id", right_index=True)
    return df


def make_train_user2producers(
    df: pd.DataFrame,
    trained_model_callback: TrainedModels,
    user_producers_df: pd.DataFrame,
) -> pd.DataFrame:
    user2producers = (
        user_producers_df.loc[:, ["user_id", "producers"]]
        .groupby("user_id")
        .apply(lambda x: x["producers"].unique().tolist())
    )
    user2producers_model = Word2Vec(
        user2producers, vector_size=10, min_count=1, window=10
    )
    trained_model_callback["user2producers"] = user2producers_model

    def get_mean_producer_vec(producers: list[str]) -> np.ndarray:
        producer_vecs = list(map(lambda x: user2producers_model.wv[x], producers))
        return np.mean(producer_vecs, axis=0)

    user_features = dict(
        map(lambda x: (x[0], get_mean_producer_vec(x[1])), user2producers.items())
    )
    user_features_df = pd.DataFrame.from_dict(user_features, orient="index")
    user_features_df.columns = [f"user2producers_vec_{i}" for i in range(10)]
    df = pd.merge(df, user_features_df, left_on="user_id", right_index=True)
    return df


def make_test_user2producers(
    df: pd.DataFrame,
    trained_models: TrainedModels,
    user_producers_df: pd.DataFrame,
) -> pd.DataFrame:
    user2producers = (
        user_producers_df.loc[:, ["user_id", "producers"]]
        .groupby("user_id")
        .apply(lambda x: x["producers"].unique().tolist())
    )
    user2producers_model = trained_models["user2producers"]
    if user2producers_model is None:
        raise ValueError("user2producers_model is None")

    def get_mean_producer_vec(producers: list[str]) -> np.ndarray:
        producer_vecs = list(map(lambda x: user2producers_model.wv[x], producers))
        return np.mean(producer_vecs, axis=0)

    user_features = dict(
        map(lambda x: (x[0], get_mean_producer_vec(x[1])), user2producers.items())
    )
    user_features_df = pd.DataFrame.from_dict(user_features, orient="index")
    user_features_df.columns = [f"user2producers_vec_{i}" for i in range(10)]
    df = pd.merge(df, user_features_df, left_on="user_id", right_index=True)
    return df


def make_train_source_encode(
    df: pd.DataFrame,
    trained_model_callback: TrainedModels,
    all_source: pd.DataFrame,
) -> pd.DataFrame:
    if trained_model_callback["source_le"] is not None:
        raise ValueError("source_le is not None")

    le = LabelEncoder()
    le.fit(all_source)

    trained_model_callback["source_le"] = le

    df["source"] = le.transform(df["source"])
    return df


def make_test_source_encode(
    df: pd.DataFrame,
    trained_models: TrainedModels,
) -> pd.DataFrame:
    if trained_models["source_le"] is None:
        raise ValueError("source_le is None")

    le = trained_models["source_le"]
    df["source"] = le.transform(df["source"])
    return df


def make_stats_episodes_per_user(
    df: pd.DataFrame, user_episodes: pd.DataFrame
) -> pd.DataFrame:
    user_episodes["episodes"] = user_episodes["episodes"].apply(
        lambda x: int(x) if x != "Unknown" else -999
    )
    user_stats_of_episodes = user_episodes.groupby("user_id")["episodes"].agg(
        ["min", "max", "mean", "std", "median", "sum"]
    )
    user_stats_of_episodes.columns = [
        "user2episodes_" + x for x in user_stats_of_episodes.columns
    ]

    df = pd.merge(df, user_stats_of_episodes, left_on="user_id", right_index=True)
    return df


def make_frequency_of_anime(df: pd.DataFrame, anime: pd.DataFrame) -> pd.DataFrame:
    anime_frequencies = anime["anime_id"].value_counts().to_frame()
    anime_frequencies.columns = ["anime_frequency"]
    df = pd.merge(df, anime_frequencies, left_on="anime_id", right_index=True)
    return df


def make_train_rating_one_hot(
    df: pd.DataFrame, trained_model_callback: TrainedModels, all_rating: pd.DataFrame
) -> pd.DataFrame:
    if trained_model_callback["rating_encoder"] is not None:
        raise ValueError("rating_encoder is not None")

    encoder = OneHotEncoder()
    encoder.fit(all_rating)

    trained_model_callback["rating_encoder"] = encoder

    onehot = encoder.transform(df.loc[:, ["rating"]])
    encoded_df = pd.DataFrame(onehot.toarray())
    encoded_df.columns = [f"rating_{i}" for i in range(len(encoder.categories_[0]))]
    encoded_df["anime_id"] = df["anime_id"]
    df = pd.merge(df, encoded_df, on="anime_id")
    return df


def make_test_rating_one_hot(
    df: pd.DataFrame, trained_models: TrainedModels, all_rating: pd.DataFrame
) -> pd.DataFrame:
    if trained_models["rating_encoder"] is None:
        raise ValueError("rating_encoder is None")

    encoder = trained_models["rating_encoder"]

    onehot = encoder.transform(df.loc[:, ["rating"]])
    encoded_df = pd.DataFrame(onehot.toarray())
    encoded_df.columns = [f"rating_{i}" for i in range(len(encoder.categories_[0]))]
    encoded_df["anime_id"] = df["anime_id"]
    df = pd.merge(df, encoded_df, on="anime_id")
    return df


#################################
# pipeline
#################################
def pipe(
    df: pd.DataFrame, *funcs: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    for func in funcs:
        df = func(df)
    return df


def train_preprocess(
    df: pd.DataFrame,
    trained_model_callback: TrainedModels,
    all_merged: pd.DataFrame,
) -> pd.DataFrame:
    df = pipe(
        df,
        cast_episode,
        partial(
            user_anime_histories_vec,
            trained_model_callback=trained_model_callback,
            user2anime=all_merged.loc[:, ["user_id", "anime_id"]],
        ),
        partial(
            make_train_user2producers,
            trained_model_callback=trained_model_callback,
            user_producers_df=all_merged.loc[:, ["user_id", "producers"]],
        ),
        partial(
            make_train_source_encode,
            trained_model_callback=trained_model_callback,
            all_source=all_merged.loc[:, ["source"]],
        ),
        partial(
            make_stats_episodes_per_user,
            user_episodes=all_merged.loc[:, ["user_id", "episodes"]],
        ),
        partial(
            make_frequency_of_anime,
            anime=all_merged.loc[:, ["anime_id"]],
        ),
        partial(
            make_train_rating_one_hot,
            trained_model_callback=trained_model_callback,
            all_rating=all_merged.loc[:, ["rating"]],
        ),
    )
    return df


def test_preprocess(
    df: pd.DataFrame,
    trained_models: TrainedModels,
    all_merged: pd.DataFrame,
) -> pd.DataFrame:
    df = pipe(
        df,
        cast_episode,
        partial(
            make_user_anime_histories,
            trained_models=trained_models,
            user2anime=all_merged.loc[:, ["user_id", "anime_id"]],
        ),
        partial(
            make_test_user2producers,
            trained_models=trained_models,
            user_producers_df=all_merged.loc[:, ["user_id", "producers"]],
        ),
        partial(
            make_test_source_encode,
            trained_models=trained_models,
        ),
        partial(
            make_stats_episodes_per_user,
            user_episodes=all_merged.loc[:, ["user_id", "episodes"]],
        ),
        partial(
            make_frequency_of_anime,
            anime=all_merged.loc[:, ["anime_id"]],
        ),
        partial(
            make_test_rating_one_hot,
            trained_models=trained_models,
            all_rating=all_merged.loc[:, ["rating"]],
        )
    )
    return df


if __name__ == "__main__":
    from src.constants import ROOT

    features = [
        "members",
        "watching",
        "completed",
        "on_hold",
        "dropped",
        "plan_to_watch",
        "episodes",
        "watch_histories_vec_0",
        "watch_histories_vec_1",
        "watch_histories_vec_2",
        "watch_histories_vec_3",
        "watch_histories_vec_4",
        "watch_histories_vec_5",
        "watch_histories_vec_6",
        "watch_histories_vec_7",
        "watch_histories_vec_8",
        "watch_histories_vec_9",
        "user2producers_vec_0",
        "user2producers_vec_1",
        "user2producers_vec_2",
        "user2producers_vec_3",
        "user2producers_vec_4",
        "user2producers_vec_5",
        "user2producers_vec_6",
        "user2producers_vec_7",
        "user2producers_vec_8",
        "user2producers_vec_9",
        "source",
        "user2episodes_min",
        "user2episodes_max",
        "user2episodes_mean",
        "user2episodes_std",
        "user2episodes_median",
        "user2episodes_sum",
        "anime_frequency",
    ]

    train_df = pd.merge(
        pd.read_csv(ROOT / "input" / "train.csv", nrows=1000),
        pd.read_csv(ROOT / "input" / "anime.csv"),
    )
    test_df = pd.merge(
        pd.read_csv(ROOT / "input" / "test.csv", nrows=1000),
        pd.read_csv(ROOT / "input" / "anime.csv"),
    )

    trained_model_callback: TrainedModels = {
        "watch_histories": None,
        "user2producers": None,
        "source_le": None,
        "rating_encoder": None,
    }

    all_df = pd.concat([train_df, test_df])
    all_source = all_df.loc[:, ["source"]]

    df = train_preprocess(
        train_df,
        trained_model_callback=trained_model_callback,
        all_merged=all_df,
    )
    print(df.shape)
    print(df.head())

    df = df[["user_id", "anime_id"] + features]
    print(df.shape)
    print(df.iloc[20:50])

    df = test_preprocess(
        test_df,
        trained_models=trained_model_callback,
        all_merged=all_df,
    )

    df = df[["user_id", "anime_id"] + features]
    print(df.shape)
    print(df.iloc[20:50])

    print(df["anime_frequency"].value_counts())
