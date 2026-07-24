#!/usr/bin/env python3
"""Aggregate NYC Taxi daily PermOOB results by model and start hour (single directory scan)."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RESULT_DIR = Path(
    "/Users/heqiaoruan/Desktop/Desktop_Heqiao/PhD/Research/permutationTestingCovariateShift/"
    "OnlinePermOOB_Benchmark/Real_Data/Real_Data/NYC_Taxi/Daily_results"
)
BATCH = 10
REF = 1000
HOURS = (3, 11, 15)
MODELS = {"xgb": "xgb_regressor", "mlp": "mlp_regressor"}

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

FILE_RE = re.compile(
    r"^(?P<date>.+) result_(?P<batch>\d+)_(?P<ref>\d+)_(?P<model>xgb_regressor|mlp_regressor)_h(?P<hour>\d+)\.csv$"
)


def scan_groups() -> dict[tuple[str, int], list[Path]]:
    groups: dict[tuple[str, int], list[Path]] = defaultdict(list)
    model_inv = {v: k for k, v in MODELS.items()}

    for entry in os.scandir(RESULT_DIR):
        if not entry.is_file() or not entry.name.endswith(".csv"):
            continue
        m = FILE_RE.match(entry.name)
        if not m:
            continue
        if int(m.group("batch")) != BATCH or int(m.group("ref")) != REF:
            continue
        tag = model_inv.get(m.group("model"))
        if tag is None:
            continue
        hour = int(m.group("hour"))
        if hour not in HOURS:
            continue
        groups[(tag, hour)].append(Path(entry.path))

    for key in groups:
        groups[key].sort(key=lambda p: p.name)
    return groups


def load_group(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=COLS)

    values: list[np.ndarray] = []
    for path in paths:
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if "BOCPD_SUM" not in df.columns:
            continue
        row = df.reindex(columns=COLS).iloc[0].to_numpy(dtype=float)
        values.append(row)
    if not values:
        return pd.DataFrame(columns=COLS)
    return pd.DataFrame(np.vstack(values), columns=COLS)


def main() -> None:
    groups = scan_groups()
    legacy: dict[str, list[pd.DataFrame]] = {k: [] for k in MODELS}

    for model_tag in MODELS:
        for hour in HOURS:
            paths = groups.get((model_tag, hour), [])
            daily = load_group(paths)
            if daily.empty:
                stats = pd.DataFrame(columns=["stat", *COLS])
            else:
                stats = pd.DataFrame(
                    [
                        ["mean", *daily.mean().tolist()],
                        ["std", *daily.std(ddof=1).tolist()],
                    ],
                    columns=["stat", *COLS],
                )

            stats_path = RESULT_DIR / f"{model_tag}_Statistics_taxi_h{hour}.csv"
            daily_path = RESULT_DIR / f"{model_tag}_daily_taxi_h{hour}.csv"
            stats.to_csv(stats_path, index=False)
            daily.to_csv(daily_path, index=False)

            if not stats.empty:
                block = stats.copy()
                block.insert(0, "hour", hour)
                legacy[model_tag].append(block)

            print(f"{model_tag} h={hour}: {len(paths)} days -> {stats_path.name} shape={stats.shape}")

    for model_tag, frames in legacy.items():
        name = "xgb_Statistics_taxi.csv" if model_tag == "xgb" else "MLP_Statistics_taxi.csv"
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(RESULT_DIR / name, index=False)
            print(f"Wrote {name}")

    print("Done.")


if __name__ == "__main__":
    main()
