"""Pure Price Alpha Probe: Predicting 30s Mid-to-Mid Return without Execution Confounding.

Tests whether microstructure features contain pure directional alpha:
Target: mid_return_bps[30s]
Models: CatBoost Regressors across feature dimensionality arms (2169 vs Top-30 vs Curated 10).
Evaluates both Pure Mid Alpha Monotonicity and Downstream Realized Execution (IOC Net PnL).
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
        "ioc_execution.sell.training_bps[30s]",
        "ioc_execution.buy.net_bps[30s]",
        "ioc_execution.sell.net_bps[30s]",
        "ioc_execution.buy.fee_bps[30s]",
        "ioc_execution.sell.fee_bps[30s]",
    ]

    print("[1/4] Loading 35-day Train and full 19-day Tune data...")
    t0 = time.time()
    cols_to_load = list(dict.fromkeys(target_cols + all_features))
    df_train = load_dataset(train_csv, base_obj_dir, cols_to_load, max_days=35)
    df_tune = load_dataset(tune_csv, base_obj_dir, cols_to_load)

    # Filter valid rows
    valid_train = np.isfinite(df_train["mid_return_bps[30s]"])
    df_train = df_train.loc[valid_train].reset_index(drop=True)
    valid_tune = np.isfinite(df_tune["mid_return_bps[30s]"])
    df_tune = df_tune.loc[valid_tune].reset_index(drop=True)
    print(f"Loaded Train: {len(df_train)}, Tune: {len(df_tune)} in {time.time() - t0:.2f}s")

    # Univariate IC with mid_return_bps[30s]
    print("\n[2/4] Computing univariate Rank IC with mid_return_bps[30s] on train...")
    y_train_mid = df_train["mid_return_bps[30s]"].to_numpy()
    ics = {}
    for feat in all_features:
        if pd.api.types.is_numeric_dtype(df_train[feat].dtype):
            vals = pd.to_numeric(df_train[feat], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(vals) & np.isfinite(y_train_mid)
            if np.sum(m) > 1000 and np.std(vals[m]) > 1e-6:
                r, _ = spearmanr(vals[m], y_train_mid[m])
                if np.isfinite(r):
                    ics[feat] = r

    sorted_ics = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)
    top_30_features = [f for f, _ in sorted_ics[:30]]

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

    print("Top 10 features correlated with mid_return_bps[30s]:")
    for name, r in sorted_ics[:10]:
        print(f"  - {name}: IC = {r:+.4f}")

    arms = {
        "Full 2169 Features": all_features,
        "Top-30 Features": top_30_features,
        "Curated-10 Micro": curated_10,
    }

    y_train = df_train["mid_return_bps[30s]"].to_numpy(dtype=np.float32)
    y_test = df_tune["mid_return_bps[30s]"].to_numpy(dtype=np.float32)
    day_test = df_tune["day"].to_numpy()

    # Pre-extract execution payoffs for simulation
    ioc_buy_net = np.nan_to_num(df_tune["ioc_execution.buy.net_bps[30s]"].to_numpy(dtype=np.float32), nan=0.0)
    ioc_sell_net = np.nan_to_num(df_tune["ioc_execution.sell.net_bps[30s]"].to_numpy(dtype=np.float32), nan=0.0)

    results = []

    print("\n[3/4] Training CatBoost Regressors to predict 30s Mid Return...")
    for arm_name, feat_cols in arms.items():
        print(f"\n--- Arm: {arm_name} ({len(feat_cols)} features) ---")
        x_train, cat_train = model_values(df_train, feat_cols)
        x_test, _ = model_values(df_tune, feat_cols, categorical=cat_train)

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

        # Daily IC
        daily_ics = []
        for d in np.unique(day_test):
            mask = day_test == d
            if np.sum(mask) > 20 and np.std(preds_test[mask]) > 1e-6 and np.std(y_test[mask]) > 1e-6:
                r, _ = spearmanr(preds_test[mask], y_test[mask])
                if np.isfinite(r):
                    daily_ics.append(r)

        # Decile analysis on Mid Return
        df_dec = pd.DataFrame(
            {
                "pred": preds_test,
                "mid_return": y_test,
                "ioc_buy_net": ioc_buy_net,
                "ioc_sell_net": ioc_sell_net,
            }
        )
        df_dec["decile"] = pd.qcut(df_dec["pred"], 10, labels=False)
        mid_decile_means = df_dec.groupby("decile")["mid_return"].mean().to_dict()

        # Decile 9 = Buy signal (highest predicted mid return) -> evaluate ioc_buy_net
        decile_9_buy_pnl = float(df_dec.loc[df_dec["decile"] == 9, "ioc_buy_net"].mean())
        # Decile 0 = Sell signal (lowest predicted mid return) -> evaluate ioc_sell_net
        decile_0_sell_pnl = float(df_dec.loc[df_dec["decile"] == 0, "ioc_sell_net"].mean())

        # Top 1% Buy and Bottom 1% Sell
        p99 = np.percentile(preds_test, 99)
        p01 = np.percentile(preds_test, 1)
        top1_buy_pnl = float(df_dec.loc[df_dec["pred"] >= p99, "ioc_buy_net"].mean())
        top1_sell_pnl = float(df_dec.loc[df_dec["pred"] <= p01, "ioc_sell_net"].mean())
        top1_mid_move = float(df_dec.loc[df_dec["pred"] >= p99, "mid_return"].mean())
        bot1_mid_move = float(df_dec.loc[df_dec["pred"] <= p01, "mid_return"].mean())

        print(f"Fit Time: {fit_time:.2f}s | Train IC: {train_ic:+.4f} | Test IC: {test_ic:+.4f} | Overfit Delta: {train_ic - test_ic:+.4f}")
        print(f"Daily Test IC: Mean = {np.mean(daily_ics):+.4f}, Positive Days = {np.mean([x > 0 for x in daily_ics]) * 100:.1f}%")
        print("Mid Return Monotonicity (Decile 0 to 9 bps):")
        for d in range(10):
            print(f"  Decile {d}: {mid_decile_means[d]:+.4f} bps")
        print(f"Execution when trading Mid Signal:")
        print(f"  - Decile 9 Buy (Top 10%): Mid Return = {mid_decile_means[9]:+.4f} bps | IOC Net PnL = {decile_9_buy_pnl:+.4f} bps")
        print(f"  - Decile 0 Sell (Bot 10%): Mid Return = {mid_decile_means[0]:+.4f} bps | IOC Net PnL = {decile_0_sell_pnl:+.4f} bps")
        print(f"  - Top 1% Buy Signal:     Mid Return = {top1_mid_move:+.4f} bps | IOC Net PnL = {top1_buy_pnl:+.4f} bps")
        print(f"  - Bot 1% Sell Signal:    Mid Return = {bot1_mid_move:+.4f} bps | IOC Net PnL = {top1_sell_pnl:+.4f} bps")

        results.append(
            {
                "arm": arm_name,
                "features_count": len(feat_cols),
                "fit_time": fit_time,
                "train_ic": float(train_ic),
                "test_ic": float(test_ic),
                "overfit_delta": float(train_ic - test_ic),
                "mean_daily_ic": float(np.mean(daily_ics)),
                "pct_positive_days": float(np.mean([x > 0 for x in daily_ics]) * 100),
                "mid_deciles": mid_decile_means,
                "top_10pct_buy_pnl": decile_9_buy_pnl,
                "bot_10pct_sell_pnl": decile_0_sell_pnl,
                "top_1pct_buy_pnl": top1_buy_pnl,
                "bot_1pct_sell_pnl": top1_sell_pnl,
                "top_1pct_mid_move": top1_mid_move,
                "bot_1pct_mid_move": bot1_mid_move,
            }
        )

    print("\n[4/4] Writing results to disk...")
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(f"{out_dir}/pure_alpha_results.csv", index=False)
    with open(f"{out_dir}/pure_alpha_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPure alpha comparison complete! Results saved in {out_dir}/")


if __name__ == "__main__":
    main()
