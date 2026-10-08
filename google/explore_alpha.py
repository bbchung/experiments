"""Deep Alpha Exploration: Decile Monotonicity, Feature Importance, and Extreme Thresholds.

Analyzes the Top-30 CatBoost Regressor on MXF 30s.
Evaluates out-of-sample monotonic payoff across deciles and extreme percentiles (Top 1%, 2%, 5%).
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
        "ioc_execution.buy.training_bps[30s]",
        "ioc_execution.sell.training_bps[30s]",
        "ioc_execution.buy.outcome_status[30s]",
        "ioc_execution.sell.outcome_status[30s]",
    ]

    print("[1/5] Loading 35 days Train and full 19 days Tune data...")
    t0 = time.time()
    cols_to_load = list(dict.fromkeys(target_cols + all_features))
    df_train = load_dataset(train_csv, base_obj_dir, cols_to_load, max_days=35)
    df_tune = load_dataset(tune_csv, base_obj_dir, cols_to_load)

    # Filter NaN
    valid_train = np.isfinite(df_train["ioc_execution.buy.training_bps[30s]"]) & np.isfinite(df_train["ioc_execution.sell.training_bps[30s]"])
    df_train = df_train.loc[valid_train].reset_index(drop=True)
    valid_tune = np.isfinite(df_tune["ioc_execution.buy.training_bps[30s]"]) & np.isfinite(df_tune["ioc_execution.sell.training_bps[30s]"])
    df_tune = df_tune.loc[valid_tune].reset_index(drop=True)
    print(f"Loaded Train: {len(df_train)}, Tune: {len(df_tune)} in {time.time() - t0:.2f}s")

    # Select Top-30 features for buy and sell separately
    print("\n[2/5] Selecting Top-30 features by in-sample univariate Rank IC...")
    feature_ranks = {}
    for side in ["buy", "sell"]:
        y_train = df_train[f"ioc_execution.{side}.training_bps[30s]"].to_numpy()
        ics = {}
        for feat in all_features:
            if pd.api.types.is_numeric_dtype(df_train[feat].dtype):
                vals = pd.to_numeric(df_train[feat], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(vals) & np.isfinite(y_train)
                if np.sum(mask) > 1000 and np.std(vals[mask]) > 1e-6:
                    r, _ = spearmanr(vals[mask], y_train[mask])
                    if np.isfinite(r):
                        ics[feat] = r
        sorted_ics = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)
        feature_ranks[side] = sorted_ics
        print(f"Top 5 {side.upper()} features:")
        for name, score in sorted_ics[:5]:
            print(f"  - {name}: IC = {score:+.4f}")

    # Train Regressor for Buy and Sell with Top-30 features
    print("\n[3/5] Fitting CatBoost Regressors and evaluating Decile monotonicity...")
    evaluation_results = {}

    for side in ["buy", "sell"]:
        top_30 = [f for f, _ in feature_ranks[side][:30]]
        x_train, cat_train = model_values(df_train, top_30)
        x_test, _ = model_values(df_tune, top_30, categorical=cat_train)

        y_train = df_train[f"ioc_execution.{side}.training_bps[30s]"].to_numpy(dtype=np.float32)
        y_test = df_tune[f"ioc_execution.{side}.training_bps[30s]"].to_numpy(dtype=np.float32)
        day_test = df_tune["day"].to_numpy()

        train_pool = Pool(x_train, y_train, cat_features=cat_train)
        test_pool = Pool(x_test, y_test, cat_features=cat_train)

        model = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="RMSE",
            task_type="GPU",
            verbose=0,
            random_seed=1729,
        )

        t_fit = time.time()
        model.fit(train_pool)
        fit_time = time.time() - t_fit

        preds_train = model.predict(train_pool)
        preds_test = model.predict(test_pool)

        train_ic, _ = spearmanr(preds_train, y_train)
        test_ic, _ = spearmanr(preds_test, y_test)

        # Feature importances
        importances = model.get_feature_importance()
        feat_imp = sorted(zip(top_30, importances), key=lambda x: x[1], reverse=True)

        # Daily IC distribution
        daily_ics = []
        for d in np.unique(day_test):
            mask = day_test == d
            if np.sum(mask) > 20 and np.std(preds_test[mask]) > 1e-6 and np.std(y_test[mask]) > 1e-6:
                r, _ = spearmanr(preds_test[mask], y_test[mask])
                if np.isfinite(r):
                    daily_ics.append(r)

        # Decile analysis
        df_dec = pd.DataFrame({"pred": preds_test, "actual": y_test})
        df_dec["decile"] = pd.qcut(df_dec["pred"], 10, labels=False)
        decile_means = df_dec.groupby("decile")["actual"].mean().to_dict()

        # Extreme percentiles (top 1%, 2%, 5%, 10%)
        p99 = np.percentile(preds_test, 99)
        p98 = np.percentile(preds_test, 98)
        p95 = np.percentile(preds_test, 95)
        p90 = np.percentile(preds_test, 90)

        ret_p99 = float(np.mean(y_test[preds_test >= p99]))
        ret_p98 = float(np.mean(y_test[preds_test >= p98]))
        ret_p95 = float(np.mean(y_test[preds_test >= p95]))
        ret_p90 = float(np.mean(y_test[preds_test >= p90]))
        base_ret = float(np.mean(y_test))

        print(f"\n================ [{side.upper()} REGRESSOR SUMMARY] ================")
        print(f"Fit Time: {fit_time:.2f}s | Train IC: {train_ic:+.4f} | Test IC: {test_ic:+.4f}")
        print(f"Daily Test IC: Mean = {np.mean(daily_ics):+.4f}, Positive Days = {np.mean([x > 0 for x in daily_ics]) * 100:.1f}%")
        print(f"Overall Test Return: {base_ret:.4f} bps")
        print("Percentile Payoffs:")
        print(f"  - Top 10% (P90+): {ret_p90:+.4f} bps (Edge: {ret_p90 - base_ret:+.4f} bps)")
        print(f"  - Top 5%  (P95+): {ret_p95:+.4f} bps (Edge: {ret_p95 - base_ret:+.4f} bps)")
        print(f"  - Top 2%  (P98+): {ret_p98:+.4f} bps (Edge: {ret_p98 - base_ret:+.4f} bps)")
        print(f"  - Top 1%  (P99+): {ret_p99:+.4f} bps (Edge: {ret_p99 - base_ret:+.4f} bps)")

        print("\nDecile Monotonicity (Decile 0 = lowest pred, Decile 9 = highest pred):")
        for d in range(10):
            print(f"  Decile {d}: {decile_means.get(d, 0.0):+.4f} bps")

        evaluation_results[side] = {
            "train_ic": float(train_ic),
            "test_ic": float(test_ic),
            "mean_daily_ic": float(np.mean(daily_ics)),
            "pct_positive_days": float(np.mean([x > 0 for x in daily_ics]) * 100),
            "base_payoff_bps": base_ret,
            "top_10pct_payoff_bps": ret_p90,
            "top_5pct_payoff_bps": ret_p95,
            "top_2pct_payoff_bps": ret_p98,
            "top_1pct_payoff_bps": ret_p99,
            "decile_payoffs": decile_means,
            "top_features": feat_imp[:10],
        }

    # Save results
    with open(f"{out_dir}/deep_regressor_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\nResults successfully saved to {out_dir}/deep_regressor_results.json")


if __name__ == "__main__":
    main()
