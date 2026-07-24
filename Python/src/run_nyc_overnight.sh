#!/usr/bin/env bash
# Launch all NYC taxi eval jobs for overnight run (skip if already running).
set -uo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="/Users/heqiaoruan/Desktop/Desktop_Heqiao/PhD/Research/permutationTestingCovariateShift/OnlinePermOOB_Benchmark"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

JOBS=(
  "NYC_Taxi_eval_results.sh:nyc_main_mlp_xgb"
  "NYC_Taxi_eval_xgb_h3.sh:nyc_xgb_h3"
  "NYC_Taxi_eval_xgb_h11.sh:nyc_xgb_h11"
  "NYC_Taxi_eval_xgb_h15.sh:nyc_xgb_h15"
  "NYC_Taxi_eval_xgb_h3_batches.sh:nyc_xgb_h3_batches"
  "NYC_Taxi_eval_xgb_h1115_batches.sh:nyc_xgb_h1115_batches"
  "NYC_Taxi_eval_xgb_h15_batches.sh:nyc_xgb_h15_batches"
)

for entry in "${JOBS[@]}"; do
  script="${entry%%:*}"
  tag="${entry##*:}"
  if pgrep -f "bash ${script}" >/dev/null 2>&1; then
    echo "ALREADY RUNNING: $script"
    continue
  fi
  echo "START: $script -> $LOGDIR/${tag}.log"
  nohup bash "$DIR/$script" >> "$LOGDIR/${tag}.log" 2>&1 &
done

echo ""
echo "=== overnight jobs $(date) ===" | tee -a "$LOGDIR/overnight_status.log"
pgrep -fl "NYC_Taxi_eval|nyc_taxi_data_evaluations" | tee -a "$LOGDIR/overnight_status.log" || true
