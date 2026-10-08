"""Selective Gated Taker Strategy on MXF 30s.

Explores if filtering by spread tightness and extreme Regressor prediction thresholds
can produce a viable trading strategy with:
- Net PnL > 0 (bps per trade)
- Daily Sharpe Ratio > 1.5
- Stable cumulative equity curve across 19 out-of-sample tune days.
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
from catboost import CatBoostRegressor, Pool
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
        "ioc_execution.buy.training_bps[30s]",
        "ioc_execution.sell.training_bps[30s]",
        "ioc_execution.buy.net_bps[30s]",
        "ioc_execution.sell.net_bps[30s]",
        "ioc_execution.buy.fee_bps[30s]",
        "ioc_execution.sell.fee_bps[30s]",
        "mid_return_bps[30s]",
    ]

    gate_cols = [
        "SpreadState.0.spread_ticks.0",
        "BookClockFlow.0.spread_ticks_mean.0",
        "TouchDepthRatio.0.depth_ratio.0",
    ]
    # Check which gate cols are in all_features
    gate_cols = [c for c in gate_cols if c in all_features]

    print("[1/4] Loading Train and Tune data...")
    cols_to_load = list(dict.fromkeys(target_cols + all_features + gate_cols))
    df_train = load_dataset(train_csv, base_obj_dir, cols_to_load, max_days=35)
    df_tune = load_dataset(tune_csv, base_obj_dir, cols_to_load)

    valid_train = np.isfinite(df_train["ioc_execution.buy.training_bps[30s]"])
    df_train = df_train.loc[valid_train].reset_index(drop=True)
    valid_tune = np.isfinite(df_tune["ioc_execution.buy.training_bps[30s]"])
    df_tune = df_tune.loc[valid_tune].reset_index(drop=True)

    # Top-30 features by in-sample univariate IC
    print("\n[2/4] Selecting Top-30 features for Buy...")
    y_train = df_train["ioc_execution.buy.training_bps[30s]"].to_numpy()
    ics = {}
    for feat in all_features:
        if pd.api.types.is_numeric_dtype(df_train[feat].dtype):
            vals = pd.to_numeric(df_train[feat], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(y_train)
            if np.sum(m) > 1000 and np.std(vals[m]) > 1e-6:
                r, _ = spearmanr(vals[m], y_train[m])
                if np.isfinite(r):
                    ics[feat] = abs(r)
    top_30 = [k for k, _ in sorted(ics.items(), key=lambda x: x[1], reverse=True)[:30]]

    # Train Regressor
    print("\n[3/4] Fitting Top-30 CatBoost Regressor...")
    x_train, cat_train = model_values(df_train, top_30)
    x_test, _ = model_values(df_tune, top_30, categorical=cat_train)

    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="RMSE",
        task_type="GPU",
        verbose=0,
        random_seed=1729,
    )
    model.fit(Pool(x_train, y_train, cat_features=cat_train))

    preds = model.predict(Pool(x_test, cat_features=cat_train))
    df_tune["pred"] = preds

    # Extract execution outcomes
    net_pnl = np.nan_to_num(df_tune["ioc_execution.buy.net_bps[30s]"].to_numpy(), nan=0.0)
    fee_bps = np.nan_to_num(df_tune["ioc_execution.buy.fee_bps[30s]"].to_numpy(), nan=0.0)
    filled = fee_bps > 0
    df_tune["net_pnl"] = net_pnl
    df_tune["filled"] = filled

    # Spread column if available
    spread_col = None
    for cand in ["SpreadState.0.spread_ticks.0", "BookClockFlow.0.spread_ticks_mean.0"]:
        if cand in df_tune.columns:
            spread_col = cand
            break

    print("\n[4/4] Simulating Selective Gated Taker Strategies...")
    threshold_quantiles = [0.90, 0.95, 0.98, 0.99, 0.995, 0.998]

    strategy_results = []

    for q in threshold_quantiles:
        threshold = np.percentile(preds, q * 100)
        base_mask = df_tune["pred"] >= threshold

        configs = [("No Gate", base_mask)]
        if spread_col is not None:
            # Gate: tight spread only (spread <= 1.05 tick)
            tight_spread_mask = base_mask & (df_tune[spread_col] <= 1.05)
            configs.append(("Tight Spread Only (<=1 tick)", tight_spread_mask))

        for name, mask in configs:
            sub = df_tune.loc[mask]
            trade_count = len(sub)
            if trade_count == 0:
                continue

            filled_sub = sub.loc[sub["filled"]]
            fill_rate = len(filled_sub) / trade_count if trade_count > 0 else 0.0

            mean_net_pnl = float(sub["net_pnl"].mean())
            filled_mean_net_pnl = float(filled_sub["net_pnl"].mean()) if len(filled_sub) > 0 else 0.0
            win_rate = float(np.mean(sub["net_pnl"] > 0)) if trade_count > 0 else 0.0

            # Daily PnL and Sharpe
            daily_pnl = sub.groupby("day")["net_pnl"].sum()
            # If day has 0 trades, pnl is 0
            unique_days = np.unique(df_tune["day"])
            daily_full = pd.Series(0.0, index=unique_days)
            daily_full.update(daily_pnl)

            mean_daily = daily_full.mean()
            std_daily = daily_full.std()
            sharpe = float((mean_daily / std_daily) * np.sqrt(252)) if std_daily > 1e-6 else 0.0

            strategy_results.append(
                {
                    "threshold_quantile": q,
                    "gate_type": name,
                    "total_signals": int(trade_count),
                    "signals_per_day": round(trade_count / len(unique_days), 1),
                    "fill_rate": round(fill_rate * 100, 1),
                    "mean_net_bps": round(mean_net_pnl, 4),
                    "filled_net_bps": round(filled_mean_net_pnl, 4),
                    "win_rate_pct": round(win_rate * 100, 1),
                    "cum_pnl_bps": round(float(daily_full.sum()), 2),
                    "daily_sharpe": round(sharpe, 2),
                }
            )

    df_out = pd.DataFrame(strategy_results)
    print("\n================ SELECTIVE TAKER BACKTEST RESULTS ================")
    print(df_out.to_string(index=False))

    df_out.to_csv(f"{out_dir}/selective_taker_results.csv", index=False)
    with open(f"{out_dir}/selective_taker_results.json", "w") as f:
        json.dump(strategy_results, f, indent=2)

    print(f"\nSaved results to {out_dir}/selective_taker_results.csv")


if __name__ == "__main__":
    main()
