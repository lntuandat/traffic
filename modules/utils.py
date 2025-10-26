from pyspark.sql import SparkSession
import argparse
import os
import time
import json

METRICS_DIR = "metrics"

def build_argparser():
    p = argparse.ArgumentParser(description="Traffic Big Data HDFS Pipeline")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--timecol", default="timestamp")
    p.add_argument("--has_header", default="true")
    p.add_argument("--delimiter", default=",")
    p.add_argument("--kmin", type=int, default=3)
    p.add_argument("--kmax", type=int, default=3)
    p.add_argument("--horizon_min", type=int, default=15)
    p.add_argument("--freq_min", type=int, default=5)
    p.add_argument("--plots_local", default="./plots")
    return p

def start_spark():
    spark = (
        SparkSession.builder
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .appName("TrafficBigDataApp")
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "4g")
        .config("spark.driver.bindAddress", os.environ.get("SPARK_DRIVER_BIND_ADDRESS", "127.0.0.1"))
        .config("spark.driver.host", os.environ.get("SPARK_DRIVER_HOST", "127.0.0.1"))
    .getOrCreate()
)
    log_level = os.environ.get("SPARK_LOG_LEVEL", "INFO").upper()
    spark.sparkContext.setLogLevel(log_level)
    return spark

def write_pipeline_metrics(durations, metadata=None, path=None):
    """
    Ghi thời gian chạy pipeline ra định dạng Prometheus + JSON để Grafana đọc.
    durations: dict tên_bước -> giây
    metadata: dict thông tin thêm (ví dụ horizon, timestamp)
    """
    os.makedirs(METRICS_DIR, exist_ok=True)
    timestamp = int(time.time())
    metrics_path = path or os.path.join(METRICS_DIR, "pipeline_metrics.prom")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f'traffic_pipeline_last_run_timestamp_seconds {timestamp}\n')
        total = durations.get("total", sum(durations.values()))
        f.write(f'traffic_pipeline_total_duration_seconds {total:.2f}\n')
        for name, value in durations.items():
            f.write(f'traffic_step_duration_seconds{{step="{name}"}} {value:.2f}\n')

    json_path = os.path.join(METRICS_DIR, "pipeline_metrics.json")
    payload = {
        "timestamp": timestamp,
        "durations": durations,
        "metadata": metadata or {}
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(payload, jf, indent=2)
