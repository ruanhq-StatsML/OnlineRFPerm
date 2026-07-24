
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from benchmark_config import MODEL_REGISTRY
from onlinePermOOB import onlinePermOOB_wholedf

COLS = [
    "BOCPD_SUM", "BOCPD_bcpd_2", "BOCPD_bcpd_1", "BOCPD_bcpd_3",
    "EWMA_SUM", "EWMA_3", "EWMA_2", "EWMA_1",
    "TS_SUM", "TS_3", "TS_2", "TS_1",
    "ADWIN_SUM", "ADWIN_3", "ADWIN_2", "ADWIN_1",
    "HDMMA_SUM", "HDMMA_3", "HDMMA_2", "HDMMA_1",
    "ECDDWT_SUM", "ECDDWT_3", "ECDDWT_2",
    "HalfSpaceTrees_SUM", "HalfSpaceTrees_1", "HalfSpaceTrees_2", "HalfSpaceTrees_3",
    "martingale_SUM", "martingale2_SUM", "martingale_1", "martingale_2", "martingale_3",
    "martingale2_1", "martingale2_2", "martingale2_3",
    "fix_SUM", "fix_1", "fix_2", "fix_3",
    "addis_SUM", "addis_1", "addis_2", "addis_3",
    "saffron_SUM", "saffron_1", "saffron_2", "saffron_3",
    "DDM_SUM", "DDM_1", "DDM_2", "DDM_3",
    "STEPD_SUM", "STEPD_1", "STEPD_2", "STEPD_3",
    "HDDMA_SUM", "HDDMA_1", "HDDMA_2", "HDDMA_3",
    "ECDDWT_1",
]


def main():
    ap = argparse.ArgumentParser(description="Online Extension of the PermOOB - Taxi Dataset")
    ap.add_argument("--date", type=str, default="2013-01-10")
    ap.add_argument("--start-hour", type=int, default=5)
    ap.add_argument("--model", type=str, default="rf_regressor")
    ap.add_argument("--ref-batch-size", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--min-rows", type=int, default=2000)
    ap.add_argument("--max-rows", type=int, default=10000)
    args = ap.parse_args()

    slice_dir = Path(
        os.environ.get(
            "NYC_TAXI_SLICE",
            "Real_Data/Real_Data/NYC_Taxi/Daily_Slice",
        )
    )
    dataset_path = slice_dir / f"{args.date} pickup_df.csv"
    np.random.seed(args.seed)
    df = pd.read_csv(dataset_path)
    n_hours = args.start_hour
    df["pickup_dt"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["pickup_hour"] = df["pickup_dt"].dt.hour
    df_hour = df[df["pickup_hour"] >= n_hours]
    if len(df_hour) > 0:
        df = df_hour
    if df.shape[0] <= args.min_rows:
        print(f"Skip {args.date}: {df.shape[0]} rows (min_rows={args.min_rows})", flush=True)
        return 0

    x_cols = [
        "PULocationID", "DOLocationID", "passenger_count",
        "trip_distance", "RatecodeID", "payment_type",
    ]
    target_col = "total_amount"
    df_x = np.array(df[x_cols])
    df_y = np.array(df[target_col]).reshape(-1, 1)
    df = np.hstack([df_x, df_y])
    df = df[: args.max_rows, :]

    result1 = onlinePermOOB_wholedf(
        df,
        model_registry=MODEL_REGISTRY,
        model_m=args.model,
        ref_batch_size=args.ref_batch_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    out_dir = Path(
        os.environ.get(
            "NYC_TAXI_OUT",
            "Real_Data/Real_Data/NYC_Taxi/Daily_results",
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{args.date} result_{args.batch_size}_{args.ref_batch_size}_"
        f"{args.model}_h{args.start_hour}.csv"
    )
    pd.DataFrame(result1, columns=COLS, index=["1"]).to_csv(out_path, index=False)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
