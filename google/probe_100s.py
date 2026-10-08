"""Multi-Horizon Microstructure Alpha: Empirical Evaluation on MXF 100s Horizon.

Compares:
1. Dimensionality ablation (2185 vs Top-30 vs Curated 10) on 100s Continuous Regressors.
2. Monotonicity and Net PnL in Top Decile / Top 1% after transaction costs.
3. Mid-to-Mid price drift vs fixed friction over 100 seconds.
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
    artifact_yaml = f"{root}/AstraResearch/runs/pool-mxf/objects/50e47d47a8d6cab69c0019fd1b9a191bbca16ea040af95f5acf01be762bcdf8e/artifact.yaml"
    train_csv = f"{root}/AstraResearch/runs/pool-mxf/objects/14aef47afdc6ba7999f78318395026825325d9bd80dfc19b2cf3db5ec0441715/partitions.csv"
    tune_csv = f"{root}/AstraResearch/runs/pool-mxf/objects/9b513b7dc21f5961fef8d08bbf42310a6ff367ecf45d3a95682383402584f991/partitions.csv"
    base_obj_dir = f"{root}/AstraResearch/runs/pool-mxf/objects"
    out_dir = f"{root}/research/experiments/google/results"
    os.makedirs(out_dir, exist_ok=True)

    with open(artifact_yaml) as f:
        all_features = yaml.safe_load(f)["metadata"]["features"]

    target_cols = [
        "ioc_execution.buy.training_bps[100s]",
        "ioc_execution.sell.training_bps[100s]",
        "ioc_execution.buy.net_bps[100s]",
        "ioc_execution.sell.net_bps[100s]",
        "ioc_execution.buy.fee_bps[100s]",
        "ioc_execution.sell.fee_bps[100s]",
        "mid_return_bps[100s]",
    ]

    print("[1/4] Loading 35-day Train and full 19-day Tune data for 100s horizon...")
    t0 = time.time()
    cols_to_load = list(dict.fromkeys(target_cols + all_features))
    df_train = load_dataset(train_csv, base_obj_dir, cols_to_load, max_days=35)
    df_tune = load_dataset(tune_csv, base_obj_dir, cols_to_load)

    # Filter NaN
    valid_train = np.isfinite(df_train["ioc_execution.buy.training_bps[100s]"]) & np.isfinite(df_train["ioc_execution.sell.training_bps[100s]"])
    df_train = df_train.loc[valid_train].reset_index(drop=True)
    valid_tune = np.isfinite(df_tune["ioc_execution.buy.training_bps[100s]"]) & np.isfinite(df_tune["ioc_execution.sell.training_bps[100s]"])
    df_tune = df_tune.loc[valid_tune].reset_index(drop=True)
    print(f"Loaded Train: {len(df_train)}, Tune: {len(df_tune)} in {time.time() - t0:.2f}s")

    # Select Top-30 features for 100s buy by univariate rank IC
    print("\n[2/4] Selecting Top-30 features by in-sample univariate IC with training_bps[100s]...")
    y_train_buy = df_train["ioc_execution.buy.training_bps[100s]"].to_numpy()
    ics = {}
    for feat in all_features:
        if pd.api.types.is_numeric_dtype(df_train[feat].dtype):
            vals = pd.to_numeric(df_train[feat], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(y_train_buy)
            if np.sum(m) > 1000 and np.std(vals[m]) > 1e-6:
                r, _ = spearmanr(vals[m], y_train_buy[m])
                if np.isfinite(r):
                    ics[feat] = abs(r)

    sorted_ics = sorted(ics.items(), key=lambda x: x[1], reverse=True)
    top_30_features = [k for k, _ in sorted_ics[:30]]
    print("Top 5 features for 100s buy:")
    for name, r in sorted_ics[:5]:
        print(f"  - {name}: IC = {r:.4f}")

    curated_candidates = [
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
    curated_10 = [f for f in curated_candidates if f in all_features]
    if not curated_10:
        curated_10 = top_30_features[:10]

    arms = {
        "Full 2185 Features": all_features,
        "Top-30 Features": top_30_features,
        "Curated-10 Micro": curated_10,
    }

    results = []

    print("\n[3/4] Evaluating CatBoost Regressors on 100s Horizon...")
    for arm_name, feat_cols in arms.items():
        print(f"\n--- Testing {arm_name} ({len(feat_cols)} features) ---")
        x_train, cat_train = model_values(df_train, feat_cols)
        x_test, _ = model_values(df_tune, feat_cols, categorical=cat_train)

        for side in ["buy", "sell"]:
            y_train = df_train[f"ioc_execution.{side}.training_bps[100s]"].to_numpy(dtype=np.float32)
            y_test = df_tune[f"ioc_execution.{side}.training_bps[100s]"].to_numpy(dtype=np.float32)
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
            overfit_delta = train_ic - test_ic

            # Daily IC
            daily_ics = []
            for d in np.unique(day_test):
                mask = day_test == d
                if np.sum(mask) > 20 and np.std(preds_test[mask]) > 1e-6 and np.std(y_test[mask]) > 1e-6:
                    r, _ = spearmanr(preds_test[mask], y_test[mask])
                    if np.isfinite(r):
                        daily_ics.append(r)

            # Decile payoffs
            df_dec = pd.DataFrame({"pred": preds_test, "actual": y_test})
            df_dec["decile"] = pd.qcut(df_dec["pred"], 10, labels=False)
            decile_means = df_dec.groupby("decile")["actual"].mean().to_dict()

            p99 = np.percentile(preds_test, 99)
            p90 = np.percentile(preds_test, 90)
            top10_payoff = float(np.mean(y_test[preds_test >= p90]))
            top1_payoff = float(np.mean(y_test[preds_test >= p99]))
            base_payoff = float(np.mean(y_test))

            print(
                f"[{side.upper()}] Fit: {fit_time:.2f}s | "
                f"Train IC: {train_ic:+.4f} | Test IC: {test_ic:+.4f} | Delta: {overfit_delta:+.4f} | "
                f"Daily IC: {np.mean(daily_ics):+.4f} ({np.mean([x > 0 for x in daily_ics]) * 100:.1f}%+) | "
                f"Top 10% PnL: {top10_payoff:+.4f} bps (Base: {base_payoff:+.4f}) | Top 1% PnL: {top1_payoff:+.4f} bps"
            )

            results.append(
                {
                    "horizon": "100s",
                    "arm": arm_name,
                    "features_count": len(feat_cols),
                    "side": side,
                    "fit_time": round(fit_time, 2),
                    "train_ic": round(float(train_ic), 4),
                    "test_ic": round(float(test_ic), 4),
                    "overfit_delta": round(float(overfit_delta), 4),
                    "mean_daily_ic": round(float(np.mean(daily_ics)), 4),
                    "pct_positive_days": round(float(np.mean([x > 0 for x in daily_ics]) * 100), 1),
                    "base_payoff_bps": round(base_payoff, 4),
                    "top10_payoff_bps": round(top10_payoff, 4),
                    "top1_payoff_bps": round(top1_payoff, 4),
                    "decile_payoffs": {str(k): round(float(v), 4) for k, v in decile_means.items()},
                }
            )

    print("\n[4/4] Saving 100s results...")
    df_res = pd.DataFrame(results)
    df_res.to_csv(f"{out_dir}/probe_100s_results.csv", index=False)
    with open(f"{out_dir}/probe_100s_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Complete!")


if __name__ == "__main__":
    main()
