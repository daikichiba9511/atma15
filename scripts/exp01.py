from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

from src.constants import ROOT
from src.preprocesses import TrainedModels, test_preprocess, train_preprocess
from src.utils import Logger, add_file_handler, get_called_time_str, get_stream_logger

logger: Final[Logger] = get_stream_logger(20)
EXEC_TIME: Final[str] = get_called_time_str()

DESCRIPTION: Final[
    str
] = """
# exp01

Description
-----------

- Baseline

"""


@dataclass
class Config:
    # -- General
    exp_name = "exp01"
    description: str = DESCRIPTION
    output_dir: Path = ROOT / "output" / exp_name
    # -- Data
    data_path: Path = ROOT / "input"
    train_file: Path = data_path / "train.csv"
    test_file: Path = data_path / "test.csv"
    anime_file: Path = data_path / "anime.csv"
    samble_file: Path = data_path / "sample_submission.csv"
    # -- Model
    model_name: str = "model"
    # -- Training
    n_folds: int = 5
    epochs: int = 10


def create_fold(df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    _df = df.copy()
    _df["kfold"] = -1
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for fold, (_, valid_idx) in enumerate(skf.split(X=_df, y=_df["score"].to_numpy())):
        _df.loc[valid_idx, "kfold"] = fold
    return _df


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.sqrt(mean_squared_error(y_true, y_pred))


@dataclass
class TrainAssets:
    model_file_path: Path
    model: xgb.Booster
    scores: np.ndarray


def train_one_fold(
    train_merged: pd.DataFrame,
    fold: int,
    features: list[str],
    model_params: dict[str, object],
    output_dir: Path,
):
    train_data = train_merged[train_merged["kfold"] != fold]
    valid_data = train_merged[train_merged["kfold"] == fold]

    train_dmatrix = xgb.DMatrix(train_data[features], label=train_data["score"])
    valid_dmatrix = xgb.DMatrix(valid_data[features], label=valid_data["score"])

    model_xgb = xgb.train(
        params=model_params,
        dtrain=train_dmatrix,
        evals=[(valid_dmatrix, "valid"), (train_dmatrix, "train")],
        early_stopping_rounds=100,
        num_boost_round=5000,
        verbose_eval=100,
    )

    save_file_path = output_dir / f"xgb-model-{fold}.xgb"
    model_xgb.save_model(str(save_file_path))

    y_pred = model_xgb.predict(valid_dmatrix)
    scores = rmse(valid_data["score"].to_numpy(), y_pred)
    logger.info(f"{fold = }, {scores = }")

    return TrainAssets(model_file_path=save_file_path, model=model_xgb, scores=scores)


def predict(
    model: xgb.Booster, test_df: pd.DataFrame, features: list[str]
) -> np.ndarray:
    test_dmatrix = xgb.DMatrix(test_df[features])
    return model.predict(test_dmatrix)


def main() -> None:
    cfg = Config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    exec_id = f"{cfg.exp_name}-{EXEC_TIME}"
    add_file_handler(logger, filename=str(cfg.output_dir / f"{exec_id}.log"))

    train_df = pd.read_csv(cfg.train_file)
    anime_df = pd.read_csv(cfg.anime_file)
    test_df = pd.read_csv(cfg.test_file)

    train_merged = pd.merge(train_df, anime_df, on="anime_id", how="left")
    test_merged = pd.merge(test_df, anime_df, on="anime_id", how="left")

    model_params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "random_state": 42,
    }
    features = [
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
    ]

    logger.info(f"{features = }")
    logger.info(f"{model_params =}")

    train_merged = create_fold(train_merged, cfg.n_folds)
    trained_preprocess_models: TrainedModels = {"watch_histories": None}

    all_df = pd.concat([train_merged, test_merged], axis=0)
    all_user2anime = all_df.loc[:, ["user_id", "anime_id"]]
    print(all_user2anime)
    train_merged = train_preprocess(
        train_merged,
        trained_model_callback=trained_preprocess_models,
        user2anime=all_user2anime,
    )
    test_merged = test_preprocess(
        test_merged, trained_preprocess_models, user2anime=all_user2anime
    )

    def _train_one_fold_func(fold):
        return train_one_fold(
            train_merged=train_merged,
            fold=fold,
            features=features,
            model_params=model_params,
            output_dir=cfg.output_dir,
        )

    train_assets = list(map(_train_one_fold_func, range(cfg.n_folds)))
    oof_score = np.mean([train_asset.scores for train_asset in train_assets])
    logger.info(f"{oof_score = }")

    models = [train_asset.model for train_asset in train_assets]
    sub_df = pd.read_csv(cfg.samble_file)
    predict_func = partial(predict, test_df=test_merged, features=features)
    sub_df["score"] = np.mean(list(map(predict_func, models)), axis=0)
    logger.info(sub_df.head(5))
    sub_df.to_csv(cfg.output_dir / "submission.csv", index=False)
    logger.info(f"Finished : {exec_id}")


if __name__ == "__main__":
    main()
