"""Microstructure Alpha: Feature Dimensionality & Overfitting Probe on MXF 30s.

Compares out-of-sample performance across four feature dimensionality arms:
- Arm 1: Full PMQ set (2,169 features)
- Arm 2: Top-30 features by in-sample Rank IC
- Arm 3: Top-10 Curated Microstructure features
- Arm 4: Single Level-1 Imbalance feature
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


def evaluate_model(
    model: CatBoostClassifier,
    train_pool: Pool,
    test_pool: Pool,
    y_train: np.ndarray,
    y_test: np.ndarray,
    payoff_test: np.ndarray,
    day_test: np.ndarray,
) -> dict:
    train_probs = model.predict_proba(train_pool)[:, 1]
    test_probs = model.predict_proba(test_pool)[:, 1]

    train_auc = float(roc_auc_score(y_train, train_probs))
    test_auc = float(roc_auc_score(y_test, test_probs))
    overfit_delta = train_auc - test_auc

    # Pooled Rank IC with actual forward net return
    rank_ic, _ = spearmanr(test_probs, payoff_test)
    rank_ic = float(rank_ic) if np.isfinite(rank_ic) else 0.0

    # Top decile (top 10% highest predicted scores) mean payoff
    cutoff = np.percentile(test_probs, 90)
    top_mask = test_probs >= cutoff
    top_decile_payoff = float(np.mean(payoff_test[top_mask])) if np.any(top_mask) else 0.0
    overall_mean_payoff = float(np.mean(payoff_test))

    # Daily Rank IC stats
    daily_ics = []
    for d in np.unique(day_test):
        mask_d = day_test == d
        if np.sum(mask_d) > 20 and np.std(test_probs[mask_d]) > 1e-6 and np.std(payoff_test[mask_d]) > 1e-6:
            ic_d, _ = spearmanr(test_probs[mask_d], payoff_test[mask_d])
            if np.isfinite(ic_d):
                daily_ics.append(float(ic_d))

    positive_ic_pct = float(np.mean([ic > 0 for ic in daily_ics]) * 100) if daily_ics else 0.0
    mean_daily_ic = float(np.mean(daily_ics)) if daily_ics else 0.0

    return {
        "train_auc": round(train_auc, 4),
        "test_auc": round(test_auc, 4),
        "overfit_delta": round(overfit_delta, 4),
        "pooled_rank_ic": round(rank_ic, 4),
        "mean_daily_rank_ic": round(mean_daily_ic, 4),
        "positive_ic_days_pct": round(positive_ic_pct, 1),
        "top_decile_payoff_bps": round(top_decile_payoff, 4),
        "overall_mean_payoff_bps": round(overall_mean_payoff, 4),
    }


def main():
    root = "/home/bb/workspace/coco_dev"
    artifact_yaml = f"{root}/AstraResearch/runs/pool-mxf/objects/e61bf2357d7277406ee84d2f2d275c719365015d256eaaf4df195cbfae11969b/artifact.yaml"
    train_csv = f"{root}/AstraResearch/runs/pool-mxf/objects/54fe9212bf8a91df3f443c51bf8b8190bb396e1a58aa62369d0bd8683036e809/partitions.csv"
    tune_csv = f"{root}/AstraResearch/runs/pool-mxf/objects/77ba20722cb501aefe15c903e1575c731eda6d8b87d4560828f6b15db39ab9c2/partitions.csv"
    base_obj_dir = f"{root}/AstraResearch/runs/pool-mxf/objects"
    out_dir = f"{root}/research/experiments/google/results"
    os.makedirs(out_dir, exist_ok=True)

    with open(artifact_yaml) as f:
        all_2169_features = yaml.safe_load(f)["metadata"]["features"]

    print(f"Loaded selection artifact: {len(all_2169_features)} candidate features.")

    curated_10 = [
        "BookSideCapacity.0.level1_volume_imbalance.0",
        "BookSideCapacity.0.level2_gap_imbalance_ticks.0",
        "BookStructure.0.book_state_absolute_touch_imbalance.0",
        "AbsoluteMicroPriceOffset.0.absolute_micro_price_offset_spreads.0",
        "MultiLevelOfi.0.multilevel_ofi.0",
        "OfiMomentum.0.ofi_momentum.0",
        "TradeOrderImbalance.0.order_imb_buy_ratio0.0",
        "TradeOrderImbalance.0.order_imb_sell_ratio0.0",
        "BookClockFlow.0.spread_ticks_mean.0",
        "StickyPriceRealizedVolatility.0.realized_vol.0",
    ]

    target_cols = [
        "ioc_net_binary.buy[30s]",
        "ioc_net_binary.sell[30s]",
        "ioc_execution.buy.training_bps[30s]",
        "ioc_execution.sell.training_bps[30s]",
    ]

    # Load 35 days of training data (~105,000 samples) and full 19 days of tune data (~58,000 samples)
    train_days_count = 35
    print(f"\n[1/4] Loading Train Data (last {train_days_count} days) and Tune Data (all 19 days)...")
    t0 = time.time()
    cols_to_load = list(dict.fromkeys(target_cols + all_2169_features))
    df_train = load_dataset(train_csv, base_obj_dir, cols_to_load, max_days=train_days_count)
    df_tune = load_dataset(tune_csv, base_obj_dir, cols_to_load)

    # Filter out rows where binary targets are NaN (e.g. at the very end of day)
    valid_train = np.isfinite(df_train["ioc_net_binary.buy[30s]"]) & np.isfinite(df_train["ioc_net_binary.sell[30s]"])
    df_train = df_train.loc[valid_train].reset_index(drop=True)
    valid_tune = np.isfinite(df_tune["ioc_net_binary.buy[30s]"]) & np.isfinite(df_tune["ioc_net_binary.sell[30s]"])
    df_tune = df_tune.loc[valid_tune].reset_index(drop=True)
    print(f"Loaded Valid Train: {df_train.shape}, Tune: {df_tune.shape} in {time.time() - t0:.2f}s")

    # Feature selection for Arm 2: Compute in-sample univariate Rank IC on numeric train features
    print("\n[2/4] Computing univariate Rank IC on train to select Top-30 features...")
    y_payoff_buy = df_train["ioc_execution.buy.training_bps[30s]"].to_numpy()
    feature_ics = {}
    for feat in all_2169_features:
        if pd.api.types.is_numeric_dtype(df_train[feat].dtype):
            vals = pd.to_numeric(df_train[feat], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(vals) & np.isfinite(y_payoff_buy)
            if np.sum(valid) > 1000 and np.std(vals[valid]) > 1e-6:
                r, _ = spearmanr(vals[valid], y_payoff_buy[valid])
                feature_ics[feat] = abs(r) if np.isfinite(r) else 0.0

    sorted_feats = sorted(feature_ics.items(), key=lambda x: x[1], reverse=True)
    top_30_features = [f for f, _ in sorted_feats[:30]]
    print(f"Top 5 features by IC: {sorted_feats[:5]}")

    arms = {
        "Arm 1: Full 2169 Features": all_2169_features,
        "Arm 2: Top-30 Train IC": top_30_features,
        "Arm 3: Top-10 Curated Micro": curated_10,
        "Arm 4: Single L1 Imbalance": [curated_10[0]],
    }

    results = []

    print("\n[3/4] Training and Evaluating CatBoost Models across Arms...")
    for arm_name, feat_cols in arms.items():
        print(f"\n--- Testing {arm_name} ({len(feat_cols)} features) ---")

        x_train, cat_train = model_values(df_train, feat_cols)
        x_test, _ = model_values(df_tune, feat_cols, categorical=cat_train)

        for side in ["buy", "sell"]:
            y_train = df_train[f"ioc_net_binary.{side}[30s]"].to_numpy(dtype=np.int32)
            y_test = df_tune[f"ioc_net_binary.{side}[30s]"].to_numpy(dtype=np.int32)
            payoff_test = df_tune[f"ioc_execution.{side}.training_bps[30s]"].to_numpy(dtype=np.float32)
            day_test = df_tune["day"].to_numpy()

            train_pool = Pool(x_train, y_train, cat_features=cat_train)
            test_pool = Pool(x_test, y_test, cat_features=cat_train)

            model = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                nan_mode="Max",
                task_type="GPU",
                verbose=0,
                random_seed=1729,
            )

            t_train = time.time()
            model.fit(train_pool)
            dur = time.time() - t_train

            metrics = evaluate_model(model, train_pool, test_pool, y_train, y_test, payoff_test, day_test)
            metrics.update(
                {
                    "arm": arm_name,
                    "features_count": len(feat_cols),
                    "side": side,
                    "fit_time_seconds": round(dur, 2),
                }
            )
            results.append(metrics)
            print(
                f"[{side.upper()}] Fit: {dur:.1f}s | "
                f"Train AUC: {metrics['train_auc']} | "
                f"Test AUC: {metrics['test_auc']} | "
                f"Overfit Delta: {metrics['overfit_delta']} | "
                f"Rank IC: {metrics['pooled_rank_ic']} | "
                f"Top 10% Payoff: {metrics['top_decile_payoff_bps']} bps"
            )

    print("\n[4/4] Compiling Results and Saving Artifacts...")
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{out_dir}/probe_comparison.csv", index=False)
    with open(f"{out_dir}/probe_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n========================= EXPERIMENT COMPARISON SUMMARY =========================")
    print(df_res[["arm", "side", "features_count", "train_auc", "test_auc", "overfit_delta", "pooled_rank_ic", "top_decile_payoff_bps"]].to_markdown(index=False))
    print(f"\nAll artifacts safely written to: {out_dir}/")


if __name__ == "__main__":
    main()
