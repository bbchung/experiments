"""Hurdle Alpha: Predicting Fat-Tail Drift Beyond Fee Friction (> 3.0 bps).

Hypothesis:
Instead of RMSE shrinkage toward zero or confounded IOC fill labels,
predicting whether 30s Pure Mid Return exceeds the fee hurdle (> 3.0 bps)
with Top-30 features should identify the regime shifts that comfortably overcome
IOC crossing fees (1.5 bps).
"""

from __future__ import annotations

import ast
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from catboost import CatBoostClassifier, Pool
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

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
    cols_to_load = list(dict.fromkeys(target_cols + all_features))
    df_train = load_dataset(train_csv, base_obj_dir, cols_to_load, max_days=35)
    df_tune = load_dataset(tune_csv, base_obj_dir, cols_to_load)

    valid_train = np.isfinite(df_train["mid_return_bps[30s]"])
    df_train = df_train.loc[valid_train].reset_index(drop=True)
    valid_tune = np.isfinite(df_tune["mid_return_bps[30s]"])
    df_tune = df_tune.loc[valid_tune].reset_index(drop=True)

    # Feature selection: Top-30 features for mid_return_bps[30s]
    print("\n[2/4] Selecting Top-30 features for mid return...")
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

    # Sort by absolute correlation
    top_30_buy = [k for k, _ in sorted(ics.items(), key=lambda x: x[1], reverse=True)[:30]]
    top_30_sell = [k for k, _ in sorted(ics.items(), key=lambda x: x[1])[:30]]

    # Define Hurdle: 3.0 bps (exceeds 1.5 bps round trip fee)
    hurdle = 3.0
    y_buy_train = (df_train["mid_return_bps[30s]"] > hurdle).astype(int).to_numpy()
    y_buy_test = (df_tune["mid_return_bps[30s]"] > hurdle).astype(int).to_numpy()

    y_sell_train = (df_train["mid_return_bps[30s]"] < -hurdle).astype(int).to_numpy()
    y_sell_test = (df_tune["mid_return_bps[30s]"] < -hurdle).astype(int).to_numpy()

    print(f"Hurdle: {hurdle} bps | Train Buy Class 1 Rate: {np.mean(y_buy_train) * 100:.1f}% | Test Buy Rate: {np.mean(y_buy_test) * 100:.1f}%")

    # Evaluate models
    results = {}
    for side, feat_cols, y_tr, y_te in [("buy", top_30_buy, y_buy_train, y_buy_test), ("sell", top_30_sell, y_sell_train, y_sell_test)]:
        print(f"\n[3/4] Training Hurdle Classifier for {side.upper()}...")
        x_train, cat_train = model_values(df_train, feat_cols)
        x_test, _ = model_values(df_tune, feat_cols, categorical=cat_train)

        train_pool = Pool(x_train, y_tr, cat_features=cat_train)
        test_pool = Pool(x_test, y_te, cat_features=cat_train)

        clf = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            task_type="GPU",
            verbose=0,
            random_seed=1729,
        )
        t0 = time.time()
        clf.fit(train_pool)
        fit_time = time.time() - t0

        probs_train = clf.predict_proba(train_pool)[:, 1]
        probs_test = clf.predict_proba(test_pool)[:, 1]

        train_auc = float(roc_auc_score(y_tr, probs_train))
        test_auc = float(roc_auc_score(y_te, probs_test))

        # Realized IOC Net PnL
        actual_ioc_net = np.nan_to_num(df_tune[f"ioc_execution.{side}.net_bps[30s]"].to_numpy(), nan=0.0)
        actual_fee = np.nan_to_num(df_tune[f"ioc_execution.{side}.fee_bps[30s]"].to_numpy(), nan=0.0)
        filled = actual_fee > 0
        actual_mid = df_tune["mid_return_bps[30s]"].to_numpy()
        if side == "sell":
            actual_mid = -actual_mid

        rank_ic, _ = spearmanr(probs_test, actual_ioc_net)

        # Percentile evaluation
        percentiles = [90, 95, 98, 99, 99.5]
        tier_stats = {}
        for p in percentiles:
            cut = np.percentile(probs_test, p)
            mask = probs_test >= cut
            sub_ioc = actual_ioc_net[mask]
            sub_mid = actual_mid[mask]
            sub_filled = filled[mask]

            tier_stats[f"Top_{100 - p}%"] = {
                "count": int(np.sum(mask)),
                "fill_rate": round(float(np.mean(sub_filled) * 100), 1),
                "mean_mid_move_bps": round(float(np.mean(sub_mid)), 4),
                "mean_ioc_net_bps": round(float(np.mean(sub_ioc)), 4),
                "filled_ioc_net_bps": round(float(np.mean(sub_ioc[sub_filled])), 4) if np.any(sub_filled) else 0.0,
                "win_rate_pct": round(float(np.mean(sub_ioc > 0) * 100), 1),
            }

        print(f"\n================ [{side.upper()} HURDLE CLASSIFIER SUMMARY] ================")
        print(f"Fit Time: {fit_time:.2f}s | Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Delta: {train_auc - test_auc:.4f}")
        print(f"OOS Rank IC with IOC Net PnL: {rank_ic:+.4f}")
        print("\nTier Performance:")
        for tier, stats in tier_stats.items():
            print(
                f"  {tier} (n={stats['count']}): Mid Move = {stats['mean_mid_move_bps']:+.2f} bps | IOC Net = {stats['mean_ioc_net_bps']:+.2f} bps | Filled Net = {stats['filled_ioc_net_bps']:+.2f} bps | Win Rate = {stats['win_rate_pct']}%"
            )

        results[side] = {
            "train_auc": train_auc,
            "test_auc": test_auc,
            "rank_ic_with_net_pnl": float(rank_ic),
            "tiers": tier_stats,
        }

    with open(f"{out_dir}/hurdle_alpha_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults successfully saved to {out_dir}/hurdle_alpha_results.json")


if __name__ == "__main__":
    main()
