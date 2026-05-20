# gold-buy-timing-optimizer

金ETF（主に `1540.T`）の月内買いタイミング最適化を目的として、
市場データ取得・特徴量生成・目的変数生成までを行うプロジェクトです。

このREADMEでは、特に以下を整理しています。

- 元データとして取得している株価・指数（ティッカー一覧）
- `features.csv` に出力される説明変数カラムの意味

---

## 概要

`main.ipynb`（実体は `gold_etf_feature_fetcher.py` 相当のコード）を実行すると、
Yahoo Finance から複数アセットのOHLCVを取得し、特徴量と目的変数を作成して
`data/` にCSVを保存します。

出力される主なファイル:

- `raw_ohlcv.csv`（生OHLCV）
- `close_prices.csv`（終値）
- `volumes.csv`（出来高）
- `features.csv`（説明変数）
- `target.csv`（目的変数）
- `dataset_all.csv`（特徴量 + 目的変数）
- `dataset_train.csv`（学習用に目的変数欠損を除外）

---

## 元データ（株価・指数）一覧

`TICKERS` で定義された資産名（内部名）とYahooシンボルです。

### Gold / Precious metals

| 内部名 | Yahooシンボル | 内容 |
|---|---|---|
| gold_futures | GC=F | 金先物 |
| gold_etf_gld | GLD | 金ETF（米国） |
| gold_etf_iau | IAU | 金ETF（米国） |
| gold_etf_jp_1540 | 1540.T | 純金上場信託（日本） |
| gold_etf_jp_1326 | 1326.T | SPDRゴールド（東証） |
| gold_miners_gdx | GDX | 金鉱株ETF |
| junior_gold_miners_gdxj | GDXJ | 中小型金鉱株ETF |
| silver_etf_slv | SLV | 銀ETF |
| silver_miners_sil | SIL | 銀鉱株ETF |

### FX / Dollar

| 内部名 | Yahooシンボル | 内容 |
|---|---|---|
| usd_jpy | JPY=X | USD/JPY |
| dxy | DX-Y.NYB | ドルインデックス |

### US equity indices / ETFs

| 内部名 | Yahooシンボル | 内容 |
|---|---|---|
| sp500 | ^GSPC | S&P500指数 |
| spy | SPY | S&P500 ETF |
| nasdaq100 | ^NDX | NASDAQ100指数 |
| qqq | QQQ | NASDAQ100 ETF |
| dow | ^DJI | ダウ平均 |
| dia | DIA | ダウETF |
| russell2000 | ^RUT | Russell 2000指数 |
| iwm | IWM | Russell 2000 ETF |

### Volatility

| 内部名 | Yahooシンボル | 内容 |
|---|---|---|
| vix | ^VIX | VIX指数 |

### US sector ETFs

| 内部名 | Yahooシンボル |
|---|---|
| materials_xlb | XLB |
| energy_xle | XLE |
| financials_xlf | XLF |
| technology_xlk | XLK |
| utilities_xlu | XLU |
| consumer_staples_xlp | XLP |
| consumer_discretionary_xly | XLY |
| industrials_xli | XLI |
| healthcare_xlv | XLV |
| communication_xlc | XLC |
| real_estate_xlre | XLRE |

### Commodity-related ETFs / futures

| 内部名 | Yahooシンボル | 内容 |
|---|---|---|
| metals_mining_xme | XME | 金属・鉱業ETF |
| copper_miners_copx | COPX | 銅鉱株ETF |
| wti_crude | CL=F | WTI原油先物 |
| brent_crude | BZ=F | ブレント原油先物 |
| copper_futures | HG=F | 銅先物 |

### Resource stocks

| 内部名 | Yahooシンボル |
|---|---|
| freeport_fcx | FCX |
| bhp | BHP |
| rio_tinto | RIO |
| vale | VALE |

### Gold mining individual stocks

| 内部名 | Yahooシンボル |
|---|---|
| newmont_nem | NEM |
| barrick_gold | GOLD |
| agnico_eagle | AEM |
| kinross | KGC |
| gold_fields | GFI |
| anglogold | AU |
| wheaton | WPM |
| franco_nevada | FNV |
| royal_gold | RGLD |

### Financials / credit-sensitive stocks

| 内部名 | Yahooシンボル |
|---|---|
| jpmorgan | JPM |
| bank_of_america | BAC |
| goldman_sachs | GS |
| morgan_stanley | MS |
| blackrock | BLK |

### Large growth / risk-on proxies

| 内部名 | Yahooシンボル |
|---|---|
| nvidia | NVDA |
| microsoft | MSFT |
| apple | AAPL |
| amazon | AMZN |
| alphabet | GOOGL |
| meta | META |
| tesla | TSLA |

### Japan market

| 内部名 | Yahooシンボル | 内容 |
|---|---|---|
| nikkei225 | ^N225 | 日経225 |
| topix | ^TOPX | TOPIX |
| japan_bank_etf | 1615.T | 銀行業ETF（代替） |
| japan_reit_etf | 1476.T | J-REIT ETF（代替） |

---

## 説明変数（`features.csv`）のカラム整理

`features.csv` は、以下の特徴量ブロックを横結合したテーブルです。

## 1) リターン特徴量

- 生成元: `make_return_features(close)`
- 期間: `1, 3, 5, 10, 20, 60` 営業日
- 計算式: `close.pct_change(w)`
- カラム名: `{asset}_ret_{w}d`
- 例: `spy_ret_5d`

## 2) ボラティリティ特徴量

- 生成元: `make_volatility_features(close)`
- 期間: `5, 20, 60`
- 計算式: `daily_ret.rolling(w).std() * sqrt(252)`
- カラム名: `{asset}_vol_{w}d`
- 例: `gold_etf_gld_vol_20d`

## 3) 移動平均乖離率

- 生成元: `make_ma_gap_features(close)`
- 期間: `5, 20, 60, 120`
- 計算式: `close / close.rolling(w).mean() - 1`
- カラム名: `{asset}_ma_gap_{w}d`
- 例: `qqq_ma_gap_60d`

## 4) 高値/安値起点の位置情報（ドローダウン系）

- 生成元: `make_drawdown_features(close)`
- 期間: `20, 60, 120`
- 計算式:
  - 高値起点: `close / rolling_high - 1`
  - 安値起点: `close / rolling_low - 1`
- カラム名:
  - `{asset}_dd_from_{w}d_high`
  - `{asset}_up_from_{w}d_low`

## 5) 出来高特徴量

- 生成元: `make_volume_features(volume)`
- 内容:
  - 1日出来高変化率: `volume.pct_change()`
  - 出来高MA比率（5,20日）: `volume / volume.rolling(w).mean() - 1`
- カラム名:
  - `{asset}_volume_chg_1d`
  - `{asset}_volume_ratio_{w}d`
- 備考: 為替・指数などVolumeがない資産では列が存在しない場合あり

## 6) 相対価格特徴量（比率ベース）

- 生成元: `make_relative_features(close)`
- まず相対価格比 `numerator / denominator` を作成し、その上で以下を生成:
  - 比率のリターン（1,3,5,10,20日）
  - 比率のMA乖離（20,60日）
- 比率名（ベース）:
  - `gdx_div_gld`
  - `gdxj_div_gdx`
  - `slv_div_gld`
  - `xlb_div_spy`
  - `xle_div_spy`
  - `xlf_div_spy`
  - `xlk_div_spy`
  - `xlu_div_spy`
  - `xlp_div_spy`
  - `qqq_div_spy`
  - `jreit_div_topix`
  - `japan_bank_div_topix`
- カラム名:
  - `{ratio_name}_ret_{w}d`
  - `{ratio_name}_ma_gap_{w}d`

## 7) カレンダー特徴量

- 生成元: `make_calendar_features(index)`
- カラム:
  - `year`
  - `month`
  - `day`
  - `dayofweek`
  - `business_day_in_month`（月内営業日番号）
  - `business_days_in_month`（月内営業日数）
  - `remaining_business_days_in_month`（月内残営業日数）
  - `month_progress`（月内進捗率）
  - `is_month_start_area`（月初3営業日フラグ）
  - `is_month_end_area`（月末3営業日フラグ）

---

## 目的変数（参考）

`target.csv` の主ラベル:

- `y_future_lower`
  - `1`: 月末までに「今日より安い日」がある
  - `0`: 月末までに「今日より安い日」がない

買い判断スコアとしては、コード内コメントどおり次の解釈を想定:

- `buy_probability = 1 - Pr(y_future_lower = 1)`

---

## 実行方法（運用Runbook）

現在の `main.py` は、運用中心に以下2モードを主に使います。

### 1) 日次本番予測（live）

```bash
python main.py --mode live --asof-date 2026-05-14 --predict-date 2026-05-15 --n-estimators 80
```

- `asof-date` までのデータで学習
- `predict-date` を予測（※ `dataset_train.csv` 内に存在する日付が必要）
- 出力先: `data/predictions_live/`

### 2) 月次検証（monthly_backtest）

```bash
python main.py --mode monthly_backtest --train-end 2026-03-31 --test-start 2026-04-01 --test-end 2026-04-30 --n-estimators 80
```

- `train-end` までで学習
- `test-start`〜`test-end` を日次で厳密検証（各日を予測）
- 出力先: `data/backtest/`

### 出力ファイル命名

運用追跡しやすいよう、実行条件がファイル名に入ります。

- live例:
  - `live_2026-05-15_asof_2026-05-14_metrics.csv`
  - `live_2026-05-15_asof_2026-05-14_month_rank_regression_predictions.csv`
  - `live_2026-05-15_asof_2026-05-14_classification_predictions.csv`
- backtest例:
  - `bt_2026-04_trainEnd_2026-03-31_YYYYMMDD_HHMMSS_metrics.csv`
  - `bt_2026-04_trainEnd_2026-03-31_YYYYMMDD_HHMMSS_classification_predictions.csv`

---

## 注意事項

- Yahoo Finance 側仕様や市場休場の影響で、一部銘柄が取得失敗することがあります。
- 取得失敗銘柄は実行時に `Warning: Some assets were not downloaded` として表示されます。
- 比率特徴量は分母銘柄が欠落した場合、該当列が欠損中心になります。
- 学習時は `dataset_train.csv`（`y_future_lower` 非欠損）を使用してください。

---

## GitHub Actions 定期実行

このリポジトリには、定期実行用ワークフローを追加済みです。

  - `.github/workflows/daily_live.yml`
  - 平日毎朝（JST 08:15）に実行
  - 実行内容:
    1. `incremental_update_data.py` でデータ増分更新
    2. `main.py --mode live` で日次予測
    3. 予測CSVをArtifact化
    4. 更新データCSVを自動コミット

  - `.github/workflows/monthly_backtest.yml`
  - 毎月1日（JST 09:00）に実行
  - 実行内容:
    1. `incremental_update_data.py` でデータ増分更新
    2. 前月期間を自動解決して `--mode monthly_backtest` 実行
    3. backtest CSVをArtifact化

### 手動実行

GitHub の Actions タブから `workflow_dispatch` で手動実行できます。
`monthly_backtest` は手動実行時に `train_end/test_start/test_end` の上書き指定も可能です。
