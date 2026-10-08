"""Inspect Gross Return vs Fee Friction across Regressor deciles."""

import ast
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

root = "/home/bb/workspace/coco_dev"
artifact_yaml = f"{root}/AstraResearch/runs/pool-mxf/objects/e61bf2357d7277406ee84d2f2d275c719365015d256eaaf4df195cbfae11969b/artifact.yaml"
train_csv = f"{root}/AstraResearch/runs/pool-mxf/objects/54fe9212bf8a91df3f443c51bf8b8190bb396e1a58aa62369d0bd8683036e809/partitions.csv"
tune_csv = f"{root}/AstraResearch/runs/pool-mxf/objects/77ba20722cb501aefe15c903e1575c731eda6d8b87d4560828f6b15db39ab9c2/partitions.csv"
base_obj_dir = f"{root}/AstraResearch/runs/pool-mxf/objects"


def load_data(csv_path, max_days=None):
    df_meta = pd.read_csv(csv_path)
    if max_days:
        df_meta = df_meta.iloc[-max_days:]
    dfs = []
    cols = [
        "ioc_execution.buy.training_bps[30s]",
        "ioc_execution.sell.training_bps[30s]",
        "ioc_execution.buy.net_bps[30s]",
        "ioc_execution.sell.net_bps[30s]",
        "ioc_execution.buy.fee_bps[30s]",
        "ioc_execution.sell.fee_bps[30s]",
        "ioc_execution.buy.outcome_status[30s]",
        "ioc_execution.sell.outcome_status[30s]",
    ]
    with open(artifact_yaml) as f:
        all_feats = yaml.safe_load(f)["metadata"]["features"]
    for _, row in df_meta.iterrows():
        ident = ast.literal_eval(row["artifact"])["identity"]
        p = os.path.join(base_obj_dir, ident, row["path"])
        dfs.append(pq.read_table(p, columns=cols + all_feats).to_pandas())
    return pd.concat(dfs, ignore_index=True), all_feats


df_train, all_feats = load_data(train_csv, max_days=35)
df_tune, _ = load_data(tune_csv)

valid_train = np.isfinite(df_train["ioc_execution.buy.training_bps[30s]"])
df_train = df_train.loc[valid_train].reset_index(drop=True)
valid_tune = np.isfinite(df_tune["ioc_execution.buy.training_bps[30s]"])
df_tune = df_tune.loc[valid_tune].reset_index(drop=True)

# Select top-30 by IC
for side in ["buy", "sell"]:
    y_train = df_train[f"ioc_execution.{side}.training_bps[30s]"].to_numpy()
    ics = {}
    for feat in all_feats:
        if pd.api.types.is_numeric_dtype(df_train[feat].dtype):
            v = pd.to_numeric(df_train[feat], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(v) & np.isfinite(y_train)
            if np.sum(m) > 1000 and np.std(v[m]) > 1e-6:
                r, _ = spearmanr(v[m], y_train[m])
                if np.isfinite(r):
                    ics[feat] = abs(r)
    top_30 = [k for k, _ in sorted(ics.items(), key=lambda x: x[1], reverse=True)[:30]]

    x_train, cat_train = model_values(df_train, top_30)
    x_test, _ = model_values(df_tune, top_30, categorical=cat_train)

    model = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05, loss_function="RMSE", task_type="GPU", verbose=0, random_seed=1729)
    model.fit(Pool(x_train, y_train, cat_features=cat_train))

    preds = model.predict(Pool(x_test, cat_features=cat_train))

    net = df_tune[f"ioc_execution.{side}.net_bps[30s]"].to_numpy()
    fee = df_tune[f"ioc_execution.{side}.fee_bps[30s]"].to_numpy()
    # Replace NaN (unfilled) with 0 for net and fee
    net = np.nan_to_num(net, nan=0.0)
    fee = np.nan_to_num(fee, nan=0.0)
    gross = net + fee

    df_eval = pd.DataFrame({"pred": preds, "net": net, "gross": gross, "fee": fee, "filled": fee > 0})
    df_eval["decile"] = pd.qcut(df_eval["pred"], 10, labels=False)

    print(f"\n================ [{side.upper()} GROSS VS NET ANALYSIS] ================")
    summary = df_eval.groupby("decile")[["gross", "fee", "net", "filled"]].mean()
    print(summary.to_string())

    # Check top 1%
    p99 = np.percentile(preds, 99)
    top1 = df_eval[df_eval["pred"] >= p99]
    print(f"\nTop 1% stats:")
    print(f"  Gross Return: {top1['gross'].mean():+.4f} bps")
    print(f"  Fee Friction: {top1['fee'].mean():+.4f} bps")
    print(f"  Net Return:   {top1['net'].mean():+.4f} bps")
    print(f"  Fill Rate:    {top1['filled'].mean() * 100:.1f}%")
