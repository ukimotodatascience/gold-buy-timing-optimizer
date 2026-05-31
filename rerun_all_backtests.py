from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


TRAIN_DATA_PATH = Path("data/dataset_train.csv")
DEFAULT_TRAIN_WINDOW_YEARS = 5
TRADING_DAYS_PER_YEAR = 252


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-window-years",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_YEARS,
        help="Passed to main.py. Use 0 for all history.",
    )
    parser.add_argument(
        "--min-train-years",
        type=int,
        default=DEFAULT_TRAIN_WINDOW_YEARS,
        help="Skip months until this many years of prior rows exist.",
    )
    parser.add_argument("--start-month", type=str, default=None, help="YYYY-MM")
    parser.add_argument("--end-month", type=str, default=None, help="YYYY-MM")
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def parse_month(value: str | None, name: str) -> pd.Period | None:
    if value is None:
        return None
    try:
        return pd.Period(value, freq="M")
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {value}. Expected YYYY-MM") from exc


def load_valid_dates(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    drop_cols = {
        "Date",
        "price",
        "y_future_lower",
        "future_min_until_month_end",
        "future_max_drawdown_until_month_end",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]
    x = df[feature_cols].shift(1)
    month_key = df["Date"].dt.to_period("M")
    y_month_rank = df["price"].groupby(month_key).rank(
        method="average", ascending=True
    )
    valid = (
        x.notna().any(axis=1)
        & df["price"].notna()
        & y_month_rank.notna()
        & df["y_future_lower"].notna()
    )
    dates = (
        df.loc[valid, "Date"].dropna().sort_values().drop_duplicates().reset_index(drop=True)
    )
    if dates.empty:
        raise RuntimeError(f"No dates found in {path}")
    return dates


def build_command(
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    train_window_years: int,
    n_estimators: int | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "main.py",
        "--mode",
        "monthly_backtest",
        "--train-end",
        train_end.date().isoformat(),
        "--test-start",
        test_start.date().isoformat(),
        "--test-end",
        test_end.date().isoformat(),
        "--train-window-years",
        str(train_window_years),
    ]
    if n_estimators is not None:
        cmd.extend(["--n-estimators", str(n_estimators)])
    return cmd


def main() -> None:
    args = parse_args()
    start_month = parse_month(args.start_month, "start-month")
    end_month = parse_month(args.end_month, "end-month")
    min_train_rows = args.min_train_years * TRADING_DAYS_PER_YEAR

    dates = load_valid_dates(TRAIN_DATA_PATH)
    months = dates.dt.to_period("M").drop_duplicates().sort_values()

    for month in months:
        if start_month is not None and month < start_month:
            continue
        if end_month is not None and month > end_month:
            continue

        month_dates = dates[dates.dt.to_period("M") == month]
        train_dates = dates[dates < month_dates.min()]
        if len(train_dates) < min_train_rows:
            print(
                f"Skip {month}: prior_rows={len(train_dates)} "
                f"< min_train_rows={min_train_rows}"
            )
            continue

        cmd = build_command(
            train_end=train_dates.max(),
            test_start=month_dates.min(),
            test_end=month_dates.max(),
            train_window_years=args.train_window_years,
            n_estimators=args.n_estimators,
        )
        print("Running:", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
