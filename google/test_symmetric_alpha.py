"""Symmetric Directional Long-Short Fat-Tail Hurdle Strategy.

Evaluates whether using signed directional micro-structural features
creates a profitable bidirectional (Buy + Sell) trading strategy.
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
        "ioc_execution.sell.net_bps[30s]",
        "ioc_execution.sell.fee_bps[30s]",
    ]

    print("[1/4] Loading Train and Tune data...")
    cols = list(dict.fromkeys(target_cols + all_features))
    df_train = load_dataset(train_csv, base_obj_dir, cols, max_days=35)
    df_tune = load_dataset(tune_csv, base_obj_dir, cols)

    valid_train = np.isfinite(df_train["mid_return_bps[30s]"])
    df_train = df_train.loc[valid_train].reset_index(drop=True)
    valid_tune = np.isfinite(df_tune["mid_return_bps[30s]"])
    df_tune = df_tune.loc[valid_tune].reset_index(drop=True)

    # Rank features by linear signed Rank IC with mid_return_bps[30s]
    # Positive IC means feature goes UP when price goes UP
    # Negative IC means feature goes UP when price goes DOWN
    print("\n[2/4] Finding signed directional features on Train...")
    y_mid_tr = df_train["mid_return_bps[30s]"].to_numpy()
    ics = {}
    for feat in all_features:
        if pd.api.types.is_numeric_dtype(df_train[feat].dtype):
            vals = pd.to_numeric(df_train[feat], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(y_mid_tr)
            if np.sum(m) > 1000 and np.std(vals[m]) > 1e-6:
                r, _ = spearmanr(vals[m], y_mid_tr[m])
                if np.isfinite(r):
                    ics[feat] = r

    # Top-25 features for Buy: highest positive correlation with mid return
    buy_feats = [k for k, _ in sorted(ics.items(), key=lambda x: x[1], reverse=True)[:25]]
    # Top-25 features for Sell: lowest (most negative) correlation with mid return
    # Exclude non-directional features (e.g. vol, abs, exceed)
    sell_candidates = [k for k, _ in sorted(ics.items(), key=lambda x: x[1])]
    directional_sell_feats = []
    for feat in sell_candidates:
        f_lower = feat.lower()
        if not any(k in f_lower for k in ["vol", "abs", "exceed", "cv", "intensity", "count", "cnt", "rate"]):
            directional_sell_feats.append(feat)
        if len(directional_sell_feats) >= 25:
            break

    print(f"Top 5 Buy Features: {buy_feats[:5]}")
    print(f"Top 5 Directional Sell Features: {directional_sell_feats[:5]}")

    # Train Buy Hurdle: mid_return > 3.0
    # Train Sell Hurdle: mid_return < -3.0
    hurdle = 3.0
    y_tr_buy = (df_train["mid_return_bps[30s]"] > hurdle).astype(int).to_numpy()
    y_tr_sell = (df_train["mid_return_bps[30s]"] < -hurdle).astype(int).to_numpy()

    # Train Buy Model
    print("\n[3/4] Fitting Buy and Sell Directional Models...")
    x_tr_buy, cat_buy = model_values(df_train, buy_feats)
    x_te_buy, _ = model_values(df_tune, buy_feats, categorical=cat_buy)

    clf_buy = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, task_type="GPU", verbose=0, random_seed=1729)
    clf_buy.fit(Pool(x_tr_buy, y_tr_buy, cat_features=cat_buy))
    p_te_buy = clf_buy.predict_proba(Pool(x_te_buy, cat_features=cat_buy))[:, 1]
    p_tr_buy = clf_buy.predict_proba(Pool(x_tr_buy, cat_features=cat_buy))[:, 1]

    # Train Sell Model
    x_tr_sell, cat_sell = model_values(df_train, directional_sell_feats)
    x_te_sell, _ = model_values(df_tune, directional_sell_feats, categorical=cat_sell)

    clf_sell = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05, task_type="GPU", verbose=0, random_seed=1729)
    clf_sell.fit(Pool(x_tr_sell, y_tr_sell, cat_features=cat_sell))
    p_te_sell = clf_sell.predict_proba(Pool(x_te_sell, cat_features=cat_sell))[:, 1]
    p_tr_sell = clf_sell.predict_proba(Pool(x_tr_sell, cat_features=cat_sell))[:, 1]

    # Evaluate on Tune Data
    actual_buy_net = np.nan_to_num(df_tune["ioc_execution.buy.net_bps[30s]"].to_numpy(), nan=0.0)
    actual_sell_net = np.nan_to_num(df_tune["ioc_execution.sell.net_bps[30s]"].to_numpy(), nan=0.0)

    # In-Sample Thresholds
    th_buy_p995 = np.percentile(p_tr_buy, 99.5)
    th_sell_p995 = np.percentile(p_tr_sell, 99.5)

    buy_mask = p_te_buy >= th_buy_p995
    sell_mask = p_te_sell >= th_sell_p995

    print(f"\n================ BIDIRECTIONAL STRATEGY RESULTS (IN-SAMPLE P99.5 THRESHOLDS) ================")
    print(f"Buy Signals:  {np.sum(buy_mask)} trades | Mean Net PnL = {np.mean(actual_buy_net[buy_mask]):+.4f} bps | Total PnL = {np.sum(actual_buy_net[buy_mask]):+.2f} bps")
    print(f"Sell Signals: {np.sum(sell_mask)} trades | Mean Net PnL = {np.mean(actual_sell_net[sell_mask]):+.4f} bps | Total PnL = {np.sum(actual_sell_net[sell_mask]):+.2f} bps")

    # Combined Long-Short Portfolio
    df_tune["long_pnl"] = np.where(buy_mask, actual_buy_net, 0.0)
    df_tune["short_pnl"] = np.where(sell_mask, actual_sell_net, 0.0)
    df_tune["total_pnl"] = df_tune["long_pnl"] + df_tune["short_pnl"]

    daily = (
        df_tune.groupby("day")
        .agg(
            buy_trades=("long_pnl", lambda x: np.sum(x != 0)),
            sell_trades=("short_pnl", lambda x: np.sum(x != 0)),
            long_pnl=("long_pnl", "sum"),
            short_pnl=("short_pnl", "sum"),
            total_pnl=("total_pnl", "sum"),
        )
        .reset_index()
    )

    daily_ret = daily["total_pnl"]
    combined_sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 1e-6 else 0.0

    print(f"\nCombined Long-Short Total PnL: {daily['total_pnl'].sum():+.2f} bps")
    print(f"Combined Daily Sharpe Ratio:   {combined_sharpe:.2f}")
    print("\nDaily Breakdown:")
    print(daily.to_string(index=False))

    results = {
        "buy_signals": int(np.sum(buy_mask)),
        "buy_mean_net_bps": float(np.mean(actual_buy_net[buy_mask])),
        "buy_total_pnl": float(np.sum(actual_buy_net[buy_mask])),
        "sell_signals": int(np.sum(sell_mask)),
        "sell_mean_net_bps": float(np.mean(actual_sell_net[sell_mask])) if np.sum(sell_mask) > 0 else 0.0,
        "sell_total_pnl": float(np.sum(actual_sell_net[sell_mask])),
        "total_portfolio_pnl": float(daily["total_pnl"].sum()),
        "combined_sharpe": float(combined_sharpe),
    }

    with open(f"{out_dir}/symmetric_alpha_results.json", "w") as f:
        json.dump(results, f, indent=2)
    daily.to_csv(f"{out_dir}/symmetric_daily_breakdown.csv", index=False)


if __name__ == "__main__":
    main()
