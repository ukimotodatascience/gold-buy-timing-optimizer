from __future__ import annotations

import argparse
from pathlib import Path
from datetime import date, datetime
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


DATA_PATH = Path("data/dataset_train.csv")
OUTPUT_DIR = Path("data")
LIVE_OUTPUT_DIR = OUTPUT_DIR / "predictions_live"
BACKTEST_OUTPUT_DIR = OUTPUT_DIR / "backtest"
MIN_TRAIN_MONTHS = 12
DEFAULT_N_ESTIMATORS_FULL = 400
DEFAULT_N_ESTIMATORS_DAILY = 120


class Split(NamedTuple):
    train_idx: pd.Index
    test_idx: pd.Index
    label: pd.Period


def log(message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def select_non_all_nan_columns(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    usable_cols = x_train.columns[x_train.notna().any(axis=0)]
    dropped = x_train.shape[1] - len(usable_cols)
    return x_train[usable_cols], x_test[usable_cols], dropped


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def build_supervised_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    案B: 前営業日までの情報で当日を予測する。
    - X_t := raw_features shifted by 1 day
    - y_t := 当日の目的変数
    """
    y_price = df["price"].copy()
    month_key = df["Date"].dt.to_period("M")
    y_month_rank = y_price.groupby(month_key).rank(method="average", ascending=True)
    y_future_lower = df["y_future_lower"].copy()

    drop_cols = {
        "Date",
        "price",
        "y_future_lower",
        "future_min_until_month_end",
        "future_max_drawdown_until_month_end",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    x_raw = df[feature_cols]
    x = x_raw.shift(1)

    valid = (
        x.notna().any(axis=1)
        & y_price.notna()
        & y_month_rank.notna()
        & y_future_lower.notna()
    )
    return x, y_price, y_month_rank, y_future_lower, valid, df["Date"]


def build_fixed_train_daily_test_splits(
    dates: pd.Series,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> list[Split]:
    train_mask = dates <= train_end
    test_mask = (dates >= test_start) & (dates <= test_end)
    train_idx = dates.index[train_mask]
    test_dates = pd.Series(pd.to_datetime(dates.loc[test_mask]).unique()).sort_values()

    splits: list[Split] = []
    for d in test_dates:
        test_idx = dates.index[dates == d]
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append(
                Split(train_idx=train_idx, test_idx=test_idx, label=d.to_period("M"))
            )
    return splits


def parse_yyyy_mm_dd(value: str | None, name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid {name}: {value}. Expected YYYY-MM-DD")
    ts = pd.Timestamp(dt)
    if pd.isna(ts):
        raise ValueError(f"Invalid {name}: {value}. Expected YYYY-MM-DD")
    return cast(pd.Timestamp, ts)


def require_timestamp(value: object, name: str) -> pd.Timestamp:
    if value is None:
        raise ValueError(f"{name} must be a valid date")
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, date):
        ts = pd.Timestamp(datetime.combine(value, datetime.min.time()))
    elif isinstance(value, str):
        ts = parse_yyyy_mm_dd(value, name)
        if ts is None:
            raise ValueError(f"{name} must be a valid date")
    else:
        raise ValueError(f"{name} must be a valid date")

    if pd.isna(ts):
        raise ValueError(f"{name} must be a valid date")
    return cast(pd.Timestamp, ts)


def ensure_date_in_data(date_value: pd.Timestamp, dates: pd.Series, name: str) -> None:
    if not (dates == date_value).any():
        min_d = pd.to_datetime(dates.min()).date()
        max_d = pd.to_datetime(dates.max()).date()
        raise ValueError(
            f"{name}={pd.to_datetime(date_value).date()} is not in dataset dates. "
            f"Available range: {min_d} .. {max_d}"
        )


def run_month_rank_regression(
    x: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    splits: list[Split],
    n_estimators: int,
) -> tuple[dict, pd.DataFrame]:
    log("[RankRegression] Setup pipeline")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    preds = []
    y_tests = []
    pred_dates = []
    test_months = []

    total_folds = len(splits)
    log(f"[RankRegression] Start walk-forward training: total_folds={total_folds}")

    for fold_no, split in enumerate(splits, start=1):
        train_idx, test_idx, test_month = split.train_idx, split.test_idx, split.label
        log(
            f"[RankRegression] Fold {fold_no}/{total_folds} | month={test_month} | "
            f"train_rows={len(train_idx)} test_rows={len(test_idx)}"
        )
        x_train, x_test = x.loc[train_idx], x.loc[test_idx]
        x_train, x_test, dropped_cols = select_non_all_nan_columns(x_train, x_test)
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        if dropped_cols > 0:
            log(
                f"[RankRegression] Fold {fold_no}/{total_folds} dropped_all_nan_features={dropped_cols}"
            )

        model.fit(x_train, y_train)
        pred_rank = model.predict(x_test)

        preds.append(pred_rank)
        y_tests.append(y_test.values)
        pred_dates.append(dates.loc[test_idx].values)
        test_months.extend([str(test_month)] * len(test_idx))
        log(f"[RankRegression] Fold {fold_no}/{total_folds} completed")

    if not preds:
        raise RuntimeError("[RankRegression] No walk-forward folds were generated.")

    pred_all = np.concatenate(preds)
    y_all = np.concatenate(y_tests)
    date_all = np.concatenate(pred_dates)

    metrics = {
        "task": "regression_month_rank",
        "train_size": len(x),
        "test_size": len(y_all),
        "walkforward_folds": len(set(test_months)),
        "rmse": float(np.sqrt(mean_squared_error(y_all, pred_all))),
        "mae": float(mean_absolute_error(y_all, pred_all)),
    }
    pred_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(date_all),
            "test_month": test_months,
            "actual_month_rank": y_all,
            "pred_month_rank": pred_all,
        },
    )
    pred_df = pred_df.set_index("Date").sort_index()
    log("[RankRegression] Completed")
    return metrics, pred_df


def run_classification(
    x: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    splits: list[Split],
    n_estimators: int,
) -> tuple[dict, pd.DataFrame]:
    log("[Classification] Setup pipeline")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )
    pred_cls_all = []
    pred_prob_all = []
    y_tests = []
    pred_dates = []
    test_months = []

    total_folds = len(splits)
    log(f"[Classification] Start walk-forward training: total_folds={total_folds}")

    for fold_no, split in enumerate(splits, start=1):
        train_idx, test_idx, test_month = split.train_idx, split.test_idx, split.label
        log(
            f"[Classification] Fold {fold_no}/{total_folds} | month={test_month} | "
            f"train_rows={len(train_idx)} test_rows={len(test_idx)}"
        )
        x_train, x_test = x.loc[train_idx], x.loc[test_idx]
        x_train, x_test, dropped_cols = select_non_all_nan_columns(x_train, x_test)
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]
        if dropped_cols > 0:
            log(
                f"[Classification] Fold {fold_no}/{total_folds} dropped_all_nan_features={dropped_cols}"
            )

        model.fit(x_train, y_train)
        pred_cls = model.predict(x_test)
        pred_prob = model.predict_proba(x_test)[:, 1]

        pred_cls_all.append(pred_cls)
        pred_prob_all.append(pred_prob)
        y_tests.append(y_test.values)
        pred_dates.append(dates.loc[test_idx].values)
        test_months.extend([str(test_month)] * len(test_idx))
        log(f"[Classification] Fold {fold_no}/{total_folds} completed")

    if not pred_cls_all:
        raise RuntimeError("[Classification] No walk-forward folds were generated.")

    pred_cls_all = np.concatenate(pred_cls_all)
    pred_prob_all = np.concatenate(pred_prob_all)
    y_all = np.concatenate(y_tests).astype(int)
    date_all = np.concatenate(pred_dates)

    if len(np.unique(y_all)) < 2:
        roc_auc = float("nan")
        log("[Classification] ROC-AUC skipped (only one class in y_true)")
    else:
        roc_auc = float(roc_auc_score(y_all, pred_prob_all))

    metrics = {
        "task": "classification_month_low",
        "train_size": len(x),
        "test_size": len(y_all),
        "walkforward_folds": len(set(test_months)),
        "accuracy": float(accuracy_score(y_all, pred_cls_all)),
        "roc_auc": roc_auc,
    }
    pred_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(date_all),
            "test_month": test_months,
            "actual_y_future_lower": y_all,
            "pred_prob_y_future_lower": pred_prob_all,
            "pred_y_future_lower": pred_cls_all,
            "pred_prob_today_is_month_low": 1 - pred_prob_all,
        },
    )
    pred_df = pred_df.set_index("Date").sort_index()
    log("[Classification] Completed")
    return metrics, pred_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["live", "monthly_backtest"],
        default="live",
        help="live=日次本番, monthly_backtest=月次検証(厳密日次再学習)",
    )
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--asof-date", type=str, default=None)
    parser.add_argument("--predict-date", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    args = parser.parse_args()

    log("=== Model training started ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Not found: {DATA_PATH}")

    log(f"Loading dataset: {DATA_PATH}")
    df = load_dataset(DATA_PATH)
    log(f"Loaded rows={len(df)}, cols={len(df.columns)}")
    x, y_price, y_month_rank, y_future_lower, valid, dates = build_supervised_dataset(
        df
    )

    x_valid = x.loc[valid]
    y_month_rank_valid = y_month_rank.loc[valid]
    y_future_lower_valid = y_future_lower.loc[valid].astype(int)
    date_valid = dates.loc[valid]
    log(
        f"Prepared supervised dataset: valid_rows={len(x_valid)}, "
        f"features={x_valid.shape[1]}"
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.mode == "live":
        asof_date = parse_yyyy_mm_dd(args.asof_date, "asof-date")
        if asof_date is None:
            max_date_raw = date_valid.max()
            if pd.isna(max_date_raw):
                raise ValueError("date_valid.max() is NaT")
            asof_date = pd.Timestamp(max_date_raw)
        asof_date = require_timestamp(asof_date, "asof-date")
        predict_date = parse_yyyy_mm_dd(args.predict_date, "predict-date")
        if predict_date is None:
            predict_date = asof_date + pd.Timedelta(days=1)
        predict_date = require_timestamp(predict_date, "predict-date")
        ensure_date_in_data(predict_date, date_valid, "predict-date")
        log(
            f"Mode=live | asof_date={asof_date.date()} predict_date={predict_date.date()}"
        )
        splits = build_fixed_train_daily_test_splits(
            date_valid,
            train_end=asof_date,
            test_start=predict_date,
            test_end=predict_date,
        )
        default_trees = DEFAULT_N_ESTIMATORS_DAILY
        output_dir = LIVE_OUTPUT_DIR
        output_prefix = f"live_{predict_date.date()}_asof_{asof_date.date()}"
    elif args.mode == "monthly_backtest":
        train_end = require_timestamp(
            parse_yyyy_mm_dd(args.train_end, "train-end"), "train-end"
        )
        test_start = require_timestamp(
            parse_yyyy_mm_dd(args.test_start, "test-start"), "test-start"
        )
        test_end = require_timestamp(
            parse_yyyy_mm_dd(args.test_end, "test-end"), "test-end"
        )
        if test_start > test_end:
            raise ValueError("test-start must be <= test-end")
        ensure_date_in_data(test_start, date_valid, "test-start")
        ensure_date_in_data(test_end, date_valid, "test-end")
        log(
            f"Mode=monthly_backtest | train_end={train_end.date()} "
            f"test_start={test_start.date()} test_end={test_end.date()}"
        )
        splits = build_fixed_train_daily_test_splits(
            date_valid,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        default_trees = DEFAULT_N_ESTIMATORS_DAILY
        output_dir = BACKTEST_OUTPUT_DIR
        output_prefix = (
            f"bt_{test_start.strftime('%Y-%m')}_trainEnd_{train_end.date()}_{run_id}"
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    n_estimators = args.n_estimators or default_trees
    log(f"Config: n_estimators={n_estimators}, folds={len(splits)}")
    if len(splits) == 0:
        raise RuntimeError(
            "No valid split was generated. Check date range and whether predict/test dates exist in dataset_train.csv."
        )

    log("Run task 1/2: regression_month_rank")
    rank_metrics, rank_pred = run_month_rank_regression(
        x_valid,
        y_month_rank_valid,
        date_valid,
        splits,
        n_estimators,
    )
    log("Run task 2/2: classification_month_low")
    cls_metrics, cls_pred = run_classification(
        x_valid,
        y_future_lower_valid,
        date_valid,
        splits,
        n_estimators,
    )

    metrics_df = pd.DataFrame([rank_metrics, cls_metrics])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{output_prefix}_metrics.csv"
    rank_path = output_dir / f"{output_prefix}_month_rank_regression_predictions.csv"
    cls_path = output_dir / f"{output_prefix}_classification_predictions.csv"

    log("Saving outputs...")
    metrics_df.to_csv(metrics_path, index=False)
    rank_pred.to_csv(rank_path)
    cls_pred.to_csv(cls_path)

    log("=== Model training completed ===")
    print(metrics_df.to_string(index=False))
    print(f"Saved: {metrics_path}")
    print(f"Saved: {rank_path}")
    print(f"Saved: {cls_path}")


if __name__ == "__main__":
    main()
