from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf


DATA_DIR = Path("data")
TARGET_ASSET = "gold_etf_jp_1540"

TICKERS: Dict[str, str] = {
    "gold_futures": "GC=F",
    "gold_etf_gld": "GLD",
    "gold_etf_iau": "IAU",
    "gold_etf_jp_1540": "1540.T",
    "gold_etf_jp_1326": "1326.T",
    "gold_miners_gdx": "GDX",
    "junior_gold_miners_gdxj": "GDXJ",
    "silver_etf_slv": "SLV",
    "silver_miners_sil": "SIL",
    "usd_jpy": "JPY=X",
    "dxy": "DX-Y.NYB",
    "sp500": "^GSPC",
    "spy": "SPY",
    "nasdaq100": "^NDX",
    "qqq": "QQQ",
    "dow": "^DJI",
    "dia": "DIA",
    "russell2000": "^RUT",
    "iwm": "IWM",
    "vix": "^VIX",
    "materials_xlb": "XLB",
    "energy_xle": "XLE",
    "financials_xlf": "XLF",
    "technology_xlk": "XLK",
    "utilities_xlu": "XLU",
    "consumer_staples_xlp": "XLP",
    "consumer_discretionary_xly": "XLY",
    "industrials_xli": "XLI",
    "healthcare_xlv": "XLV",
    "communication_xlc": "XLC",
    "real_estate_xlre": "XLRE",
    "metals_mining_xme": "XME",
    "copper_miners_copx": "COPX",
    "wti_crude": "CL=F",
    "brent_crude": "BZ=F",
    "copper_futures": "HG=F",
    "freeport_fcx": "FCX",
    "bhp": "BHP",
    "rio_tinto": "RIO",
    "vale": "VALE",
    "newmont_nem": "NEM",
    "barrick_gold": "GOLD",
    "agnico_eagle": "AEM",
    "kinross": "KGC",
    "gold_fields": "GFI",
    "anglogold": "AU",
    "wheaton": "WPM",
    "franco_nevada": "FNV",
    "royal_gold": "RGLD",
    "jpmorgan": "JPM",
    "bank_of_america": "BAC",
    "goldman_sachs": "GS",
    "morgan_stanley": "MS",
    "blackrock": "BLK",
    "nvidia": "NVDA",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "meta": "META",
    "tesla": "TSLA",
    "nikkei225": "^N225",
    "topix": "^TOPX",
    "japan_bank_etf": "1615.T",
    "japan_reit_etf": "1476.T",
}


def download_recent_ohlcv(days: int = 7) -> pd.DataFrame:
    end = datetime.now().date() + timedelta(days=1)
    start = end - timedelta(days=max(days + 7, 14))
    symbols = list(TICKERS.values())
    name_by_symbol = {v: k for k, v in TICKERS.items()}

    df = yf.download(
        symbols,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError("Failed to download recent market data.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            [(f, name_by_symbol.get(s, s)) for f, s in df.columns],
            names=["field", "asset"],
        )
    return df.sort_index()


def extract_close_volume(ohlcv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = (
        ohlcv["Close"].copy().dropna(axis=1, how="all")
        if "Close" in ohlcv.columns.get_level_values("field")
        else pd.DataFrame(index=ohlcv.index)
    )
    volume = (
        ohlcv["Volume"].copy().dropna(axis=1, how="all")
        if "Volume" in ohlcv.columns.get_level_values("field")
        else pd.DataFrame(index=ohlcv.index)
    )
    close.index = pd.to_datetime(close.index)
    volume.index = pd.to_datetime(volume.index)
    return close.sort_index(), volume.sort_index()


def merge_append(existing: pd.DataFrame, new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if existing.empty:
        merged = new.copy()
    else:
        merged = pd.concat([existing, new], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    added = (
        len(merged.index.difference(existing.index))
        if not existing.empty
        else len(merged)
    )
    return merged, added


def make_features(close: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    dt_index = pd.DatetimeIndex(pd.to_datetime(close.index))
    close_ffill = close.ffill()
    volume_ffill = volume.ffill() if not volume.empty else volume
    feats = []
    for w in (1, 3, 5, 10, 20, 60):
        d = close_ffill.pct_change(w, fill_method=None)
        d.columns = [f"{c}_ret_{w}d" for c in d.columns]
        feats.append(d)
    daily_ret = close_ffill.pct_change(fill_method=None)
    for w in (5, 20, 60):
        d = daily_ret.rolling(w).std() * np.sqrt(252)
        d.columns = [f"{c}_vol_{w}d" for c in d.columns]
        feats.append(d)
    for w in (5, 20, 60, 120):
        d = close / close.rolling(w).mean() - 1
        d.columns = [f"{c}_ma_gap_{w}d" for c in d.columns]
        feats.append(d)
    for w in (20, 60, 120):
        hi = close / close.rolling(w).max() - 1
        hi.columns = [f"{c}_dd_from_{w}d_high" for c in hi.columns]
        lo = close / close.rolling(w).min() - 1
        lo.columns = [f"{c}_up_from_{w}d_low" for c in lo.columns]
        feats.extend([hi, lo])
    if not volume.empty:
        d = volume_ffill.pct_change(fill_method=None)
        d.columns = [f"{c}_volume_chg_1d" for c in d.columns]
        feats.append(d)
        for w in (5, 20):
            d = volume / volume.rolling(w).mean() - 1
            d.columns = [f"{c}_volume_ratio_{w}d" for c in d.columns]
            feats.append(d)
    cal = pd.DataFrame(index=dt_index)
    dt_series = pd.Series(dt_index, index=dt_index)
    cal["year"] = dt_series.dt.year.values
    cal["month"] = dt_series.dt.month.values
    cal["day"] = dt_series.dt.day.values
    cal["dayofweek"] = dt_series.dt.dayofweek.values
    mk = pd.Series(dt_index.to_period("M"), index=dt_index)
    cal["business_day_in_month"] = mk.groupby(mk).cumcount() + 1
    cal["business_days_in_month"] = cal.groupby(["year", "month"])[
        "business_day_in_month"
    ].transform("max")
    cal["remaining_business_days_in_month"] = (
        cal["business_days_in_month"] - cal["business_day_in_month"]
    )
    cal["month_progress"] = cal["business_day_in_month"] / cal["business_days_in_month"]
    cal["is_month_start_area"] = (cal["business_day_in_month"] <= 3).astype(int)
    cal["is_month_end_area"] = (cal["remaining_business_days_in_month"] <= 3).astype(
        int
    )
    feats.append(cal)
    return pd.concat(feats, axis=1).replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.DataFrame, target_asset: str = TARGET_ASSET) -> pd.DataFrame:
    price = close[target_asset].copy()
    df = pd.DataFrame(index=price.index)
    df["price"] = price
    df["year_month"] = price.index.to_period("M")
    future_min = []
    for _, g in df.groupby("year_month"):
        future_min.append(g["price"][::-1].cummin()[::-1].shift(-1))
    df["future_min_until_month_end"] = pd.concat(future_min).sort_index()
    df["future_max_drawdown_until_month_end"] = (
        df["future_min_until_month_end"] / df["price"] - 1
    )
    df["y_future_lower"] = (df["future_min_until_month_end"] < df["price"]).astype(
        float
    )
    df.loc[df["future_min_until_month_end"].isna(), "y_future_lower"] = np.nan
    return df[
        [
            "price",
            "future_min_until_month_end",
            "future_max_drawdown_until_month_end",
            "y_future_lower",
        ]
    ]


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    close_path = DATA_DIR / "close_prices.csv"
    vol_path = DATA_DIR / "volumes.csv"

    existing_close = (
        pd.read_csv(close_path, index_col=0, parse_dates=True)
        if close_path.exists()
        else pd.DataFrame()
    )
    existing_vol = (
        pd.read_csv(vol_path, index_col=0, parse_dates=True)
        if vol_path.exists()
        else pd.DataFrame()
    )

    ohlcv_recent = download_recent_ohlcv(days=7)
    close_new, vol_new = extract_close_volume(ohlcv_recent)

    close_merged, added_close = merge_append(existing_close, close_new)
    vol_merged, _ = merge_append(existing_vol, vol_new)

    features = make_features(close_merged, vol_merged)
    target = make_target(close_merged)
    dataset = features.join(target, how="left")
    train_dataset = dataset.dropna(subset=["y_future_lower"]).copy()

    close_merged.to_csv(DATA_DIR / "close_prices.csv", encoding="utf-8-sig")
    vol_merged.to_csv(DATA_DIR / "volumes.csv", encoding="utf-8-sig")
    features.to_csv(DATA_DIR / "features.csv", encoding="utf-8-sig")
    target.to_csv(DATA_DIR / "target.csv", encoding="utf-8-sig")
    dataset.to_csv(DATA_DIR / "dataset_all.csv", encoding="utf-8-sig")
    train_dataset.to_csv(DATA_DIR / "dataset_train.csv", encoding="utf-8-sig")

    print(f"Added new dates to close_prices: {added_close}")
    print(f"Latest date in close_prices: {close_merged.index.max().date()}")
    print(
        "Updated: close_prices.csv, volumes.csv, features.csv, target.csv, dataset_all.csv, dataset_train.csv"
    )


if __name__ == "__main__":
    main()
