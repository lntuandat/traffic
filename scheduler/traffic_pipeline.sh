#!/usr/bin/env bash
# Cron script chạy pipeline Spark và đồng bộ kết quả về local.
#
# Ví dụ thêm vào crontab:
# 0 * * * * /home/lntuandat/traffic-bigdata/scheduler/traffic_pipeline.sh >> /home/lntuandat/traffic-bigdata/logs/pipeline.log 2>&1

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/lntuandat/traffic-bigdata}"
INPUT_PATH="${INPUT_PATH:-hdfs://localhost:9000/data/traffic/METR-LA.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-hdfs://localhost:9000/results/metr_la}"
PLOTS_LOCAL="${PLOTS_LOCAL:-$PROJECT_DIR/results}"
SPARK_CMD="${SPARK_CMD:-spark-submit}"

cd "$PROJECT_DIR"

# Dọn dữ liệu Spark cũ trước khi đồng bộ lần mới
rm -rf "$PLOTS_LOCAL"/{analysis,models,plots_kmeans,silhouette_scores,traffic_signal_plan,traffic_signal_plan_json} 2>/dev/null || true
rm -rf "$PLOTS_LOCAL"/predict_* 2>/dev/null || true
rm -f "$PLOTS_LOCAL"/prob_congestion_*.png 2>/dev/null || true

mkdir -p "$PLOTS_LOCAL"/{plots_kmeans,silhouette_scores,traffic_signal_plan,traffic_signal_plan_json,analysis,models}

start_ts=$(date +%s)

$SPARK_CMD --master local[*] main.py \
  --input "$INPUT_PATH" \
  --out "$OUTPUT_PATH" \
  --timecol timestamp \
  --has_header true \
  --plots_local "$PLOTS_LOCAL"

hdfs dfs -get -f "$OUTPUT_PATH"/predict_* "$PLOTS_LOCAL"/ 2>/dev/null || true
hdfs dfs -get -f "$OUTPUT_PATH"/plots_kmeans "$PLOTS_LOCAL"/plots_kmeans 2>/dev/null || true
hdfs dfs -get -f "$OUTPUT_PATH"/silhouette_scores "$PLOTS_LOCAL"/silhouette_scores 2>/dev/null || true
hdfs dfs -get -f "$OUTPUT_PATH"/traffic_signal_plan "$PLOTS_LOCAL"/traffic_signal_plan 2>/dev/null || true
hdfs dfs -get -f "$OUTPUT_PATH"/traffic_signal_plan_json "$PLOTS_LOCAL"/traffic_signal_plan_json 2>/dev/null || true
hdfs dfs -get -f "$OUTPUT_PATH"/analysis "$PLOTS_LOCAL"/analysis 2>/dev/null || true
hdfs dfs -get -f "$OUTPUT_PATH"/models "$PLOTS_LOCAL"/models 2>/dev/null || true

end_ts=$(date +%s)
duration=$(( end_ts - start_ts ))

echo "$(date) - Spark pipeline completed in ${duration}s"
