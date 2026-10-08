"""Sampling Stride & Horizon Overlap Ablation.

Evaluates how sampling frequency (5s overlapping vs 30s non-overlapping) affects:
1. CatBoost overfitting gap (Train IC vs Test IC)
2. Out-of-sample Rank IC
3. Downstream Top Decile PnL
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
        "mid_return_bps[30s]",
        "ioc_execution.buy.training_bps[30s]",
        "ioc_execution.buy.net_bps[30s]",
    ]

    print("[1/3] Loading Train and Tune Data...")
    cols = list(dict.fromkeys(target_cols + all_features))
    df_train = load_dataset(train_csv, base_obj_dir, cols, max_days=35)
    df_tune = load_dataset(tune_csv, base_obj_dir, cols)

    valid_train = np.isfinite(df_train["ioc_execution.buy.training_bps[30s]"])
    df_train = df_train.loc[valid_train].reset_index(drop=True)
    valid_tune = np.isfinite(df_tune["ioc_execution.buy.training_bps[30s]"])
    df_tune = df_tune.loc[valid_tune].reset_index(drop=True)

    # Top-30 features for Buy Payoff
    y_train_full = df_train["ioc_execution.buy.training_bps[30s]"].to_numpy()
    ics = {}
    for feat in all_features:
        if pd.api.types.is_numeric_dtype(df_train[feat].dtype):
            vals = pd.to_numeric(df_train[feat], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(y_train_full)
            if np.sum(m) > 1000 and np.std(vals[m]) > 1e-6:
                r, _ = spearmanr(vals[m], y_train_full[m])
                if np.isfinite(r):
                    ics[feat] = abs(r)
    top_30 = [k for k, _ in sorted(ics.items(), key=lambda x: x[1], reverse=True)[:30]]

    # Compare Strides on Train:
    # Stride 1: Every 5s (100% data, 83% overlap)
    # Stride 2: Every 10s (50% data, 66% overlap)
    # Stride 6: Every 30s (16.7% data, 0% overlap)
    strides = {
        "Stride 1 (5s grid - 83% overlap)": 1,
        "Stride 2 (10s grid - 66% overlap)": 2,
        "Stride 6 (30s grid - Non-overlapping)": 6,
    }

    results = []

    x_test, _ = model_values(df_tune, top_30)
    y_test = df_tune["ioc_execution.buy.training_bps[30s]"].to_numpy(dtype=np.float32)
    day_test = df_tune["day"].to_numpy()
    test_pool = Pool(x_test, y_test)

    print("\n[2/3] Evaluating Stride Ablation...")
    for label, stride in strides.items():
        df_sub = df_train.iloc[::stride].reset_index(drop=True)
        x_train, cat_train = model_values(df_sub, top_30)
        y_train = df_sub["ioc_execution.buy.training_bps[30s]"].to_numpy(dtype=np.float32)

        train_pool = Pool(x_train, y_train, cat_features=cat_train)

        model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            task_type="GPU",
            verbose=0,
            random_seed=1729,
        )

        t0 = time.time()
        model.fit(train_pool)
        fit_dur = time.time() - t0

        preds_train = model.predict(train_pool)
        preds_test = model.predict(test_pool)

        train_ic, _ = spearmanr(preds_train, y_train)
        test_ic, _ = spearmanr(preds_test, y_test)
        overfit_delta = train_ic - test_ic

        p90 = np.percentile(preds_test, 90)
        top10_payoff = float(np.mean(y_test[preds_test >= p90]))

        daily_ics = []
        for d in np.unique(day_test):
            mask = day_test == d
            if np.sum(mask) > 20 and np.std(preds_test[mask]) > 1e-6:
                r, _ = spearmanr(preds_test[mask], y_test[mask])
                if np.isfinite(r):
                    daily_ics.append(r)

        print(f"\n--- {label} ---")
        print(f"Train Samples: {len(df_sub):,} | Fit Time: {fit_dur:.2f}s")
        print(f"Train IC: {train_ic:+.4f} | Test IC: {test_ic:+.4f} | Overfit Delta: {overfit_delta:+.4f}")
        print(f"Mean Daily Test IC: {np.mean(daily_ics):+.4f} (Positive Days: {np.mean([x > 0 for x in daily_ics]) * 100:.1f}%)")
        print(f"Top 10% Payoff: {top10_payoff:+.4f} bps")

        results.append(
            {
                "stride_label": label,
                "stride": stride,
                "train_samples": len(df_sub),
                "fit_time": round(fit_dur, 2),
                "train_ic": round(float(train_ic), 4),
                "test_ic": round(float(test_ic), 4),
                "overfit_delta": round(float(overfit_delta), 4),
                "mean_daily_ic": round(float(np.mean(daily_ics)), 4),
                "pct_positive_days": round(float(np.mean([x > 0 for x in daily_ics]) * 100), 1),
                "top10_payoff_bps": round(top10_payoff, 4),
            }
        )

    print("\n[3/3] Saving stride ablation results...")
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{out_dir}/stride_ablation_results.csv", index=False)
    with open(f"{out_dir}/stride_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Complete!")


if __name__ == "__main__":
    main()
