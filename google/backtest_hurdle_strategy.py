"""Comprehensive Backtest & Risk Profile of the Fat-Tail Hurdle Buy Strategy.

Evaluates daily performance across the 19 out-of-sample tune days:
- Cumulative Net PnL (bps and $ equivalent)
- Daily Sharpe Ratio (annualized)
- Maximum Drawdown
- Win Rate, Profit Factor, and Trade Count
- Daily breakdown table
"""

from __future__ import annotations

import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr

sys.path.insert(0, "/home/bb/workspace/coco_dev")
from research.feature_values import model_values


def load_dataset(partitions_csv: str, base_dir: str, columns: list[str], max_days: int | None = None) -> pd.DataFrame:
    df_meta = pd.read_csv(partitions_csv)
    if max_days is not None:
        df_meta = df_meta.iloc[-max_days:]
    dfs = []
    for _, row in df_meta.iterrows():
        ident = ast.literal_eval(row["artifact"])["identity"]
        p = os.path.join(base_dir, ident, row["path"])
        t = pq.read_table(p, columns=columns)
        df_part = t.to_pandas()
        df_part["day"] = str(row["day"])
        dfs.append(df_part)
    return pd.concat(dfs, ignore_index=True)


def main():
    root = "/home/bb/workspace/coco_dev"
    artifact_yaml = f"{root}/AstraResearch/runs/pool-mxf/objects/e61bf2357d7277406ee84d2f2d275c719365015d256eaaf4df195cbfae11969b/artifact.yaml"
    train_csv = f"{root}/AstraResearch/runs/pool-mxf/objects/54fe9212bf8a91df3f443c51bf8b8190bb396e1a58aa62369d0bd8683036e809/partitions.csv"
    tune_csv = f"{root}/AstraResearch/runs/pool-mxf/objects/77ba20722cb501aefe15c903e1575c731eda6d8b87d4560828f6b15db39ab9c2/partitions.csv"
    base_obj_dir = f"{root}/AstraResearch/runs/pool-mxf/objects"
    out_dir = f"{root}/research/experiments/google/results"
    os.makedirs(out_dir, exist_ok=True)

    with open(artifact_yaml) as f:
        all_features = yaml.safe_load(f)["metadata"]["features"]

    target_cols = [
        "mid_return_bps[30s]",
        "ioc_execution.buy.net_bps[30s]",
        "ioc_execution.buy.fee_bps[30s]",
        "ioc_execution.buy.gross_cash[30s]",
    ]

    print("[1/4] Loading Train and Tune data...")
    cols_to_load = list(dict.fromkeys(target_cols + all_features))
    df_train = load_dataset(train_csv, base_obj_dir, cols_to_load, max_days=35)
    df_tune = load_dataset(tune_csv, base_obj_dir, cols_to_load)

    valid_train = np.isfinite(df_train["mid_return_bps[30s]"])
    df_train = df_train.loc[valid_train].reset_index(drop=True)
    valid_tune = np.isfinite(df_tune["mid_return_bps[30s]"])
    df_tune = df_tune.loc[valid_tune].reset_index(drop=True)

    # Select Top-30 features for positive mid_return_bps[30s]
    print("\n[2/4] Selecting Top-30 features for Buy...")
    y_mid_train = df_train["mid_return_bps[30s]"].to_numpy()
    ics = {}
    for feat in all_features:
        if pd.api.types.is_numeric_dtype(df_train[feat].dtype):
            vals = pd.to_numeric(df_train[feat], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(y_mid_train)
            if np.sum(m) > 1000 and np.std(vals[m]) > 1e-6:
                r, _ = spearmanr(vals[m], y_mid_train[m])
                if np.isfinite(r):
                    ics[feat] = r

    top_30_buy = [k for k, _ in sorted(ics.items(), key=lambda x: x[1], reverse=True)[:30]]

    # Train Hurdle Classifier
    hurdle = 3.0
    y_buy_train = (df_train["mid_return_bps[30s]"] > hurdle).astype(int).to_numpy()

    print("\n[3/4] Fitting Hurdle Classifier...")
    x_train, cat_train = model_values(df_train, top_30_buy)
    x_test, _ = model_values(df_tune, top_30_buy, categorical=cat_train)

    clf = CatBoostClassifier(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        task_type="GPU",
        verbose=0,
        random_seed=1729,
    )
    clf.fit(Pool(x_train, y_buy_train, cat_features=cat_train))

    probs_test = clf.predict_proba(Pool(x_test, cat_features=cat_train))[:, 1]
    df_tune["prob"] = probs_test

    # Extract actual execution outcomes
    actual_net = np.nan_to_num(df_tune["ioc_execution.buy.net_bps[30s]"].to_numpy(), nan=0.0)
    actual_fee = np.nan_to_num(df_tune["ioc_execution.buy.fee_bps[30s]"].to_numpy(), nan=0.0)
    df_tune["net_bps"] = actual_net
    df_tune["filled"] = actual_fee > 0

    # Backtest across multiple selectivity thresholds
    thresholds = {
        "Top 1% (P99)": np.percentile(probs_test, 99.0),
        "Top 0.5% (P99.5)": np.percentile(probs_test, 99.5),
        "Top 0.3% (P99.7)": np.percentile(probs_test, 99.7),
        "Top 0.2% (P99.8)": np.percentile(probs_test, 99.8),
    }

    all_days = np.unique(df_tune["day"])
    summary_results = []

    print("\n[4/4] Generating Detailed Daily Backtest Report...")
    for label, th in thresholds.items():
        sub = df_tune[df_tune["prob"] >= th].copy()

        daily_trades = sub.groupby("day")["net_bps"].count()
        daily_pnl = sub.groupby("day")["net_bps"].sum()

        daily_series = pd.DataFrame(index=all_days)
        daily_series["trades"] = daily_trades
        daily_series["trades"] = daily_series["trades"].fillna(0).astype(int)
        daily_series["pnl_bps"] = daily_pnl.fillna(0.0)

        # Metrics
        total_trades = int(daily_series["trades"].sum())
        total_pnl = float(daily_series["pnl_bps"].sum())
        mean_pnl_per_trade = float(sub["net_bps"].mean()) if total_trades > 0 else 0.0

        # Filled only
        sub_filled = sub[sub["filled"]]
        filled_mean = float(sub_filled["net_bps"].mean()) if len(sub_filled) > 0 else 0.0
        fill_rate = float(len(sub_filled) / total_trades * 100) if total_trades > 0 else 0.0

        # Win rate & profit factor
        wins = sub[sub["net_bps"] > 0]["net_bps"]
        losses = sub[sub["net_bps"] < 0]["net_bps"]
        win_rate = float(len(wins) / total_trades * 100) if total_trades > 0 else 0.0
        profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 1e-6 else 0.0

        # Sharpe ratio
        daily_returns = daily_series["pnl_bps"]
        sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252)) if daily_returns.std() > 1e-6 else 0.0

        # Max drawdown
        cum_pnl = daily_returns.cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_dd = float(drawdown.min())

        # Profitable days percentage
        pos_days = float(np.mean(daily_returns > 0) * 100)

        summary_results.append(
            {
                "strategy": label,
                "threshold": round(float(th), 4),
                "total_trades": total_trades,
                "trades_per_day": round(total_trades / len(all_days), 1),
                "fill_rate_pct": round(fill_rate, 1),
                "mean_net_bps_per_signal": round(mean_pnl_per_trade, 4),
                "filled_net_bps_per_trade": round(filled_mean, 4),
                "win_rate_pct": round(win_rate, 1),
                "profit_factor": round(profit_factor, 2),
                "total_cum_pnl_bps": round(total_pnl, 2),
                "daily_sharpe": round(sharpe, 2),
                "max_drawdown_bps": round(max_dd, 2),
                "profitable_days_pct": round(pos_days, 1),
            }
        )

    df_summary = pd.DataFrame(summary_results)
    print("\n================ SUMMARY STRATEGY COMPARISON ================")
    print(df_summary.to_string(index=False))

    df_summary.to_csv(f"{out_dir}/backtest_hurdle_strategy.csv", index=False)
    with open(f"{out_dir}/backtest_hurdle_strategy.json", "w") as f:
        json.dump(summary_results, f, indent=2)

    # Print daily PnL breakdown for the best strategy (Top 0.5%)
    th_best = thresholds["Top 0.5% (P99.5)"]
    sub_best = df_tune[df_tune["prob"] >= th_best]
    daily_best = (
        sub_best.groupby("day")
        .agg(
            trades=("net_bps", "count"),
            filled=("filled", "sum"),
            pnl_bps=("net_bps", "sum"),
            mean_bps=("net_bps", "mean"),
        )
        .reset_index()
    )
    print("\n================ DAILY PNL BREAKDOWN (TOP 0.5% STRATEGY) ================")
    print(daily_best.to_string(index=False))

    daily_best.to_csv(f"{out_dir}/daily_breakdown_top05pct.csv", index=False)
    print(f"\nAll backtest artifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
