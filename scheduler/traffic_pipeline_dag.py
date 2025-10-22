"""
Airflow DAG chạy pipeline Spark định kỳ.

Điều kiện:
- Airflow cài đặt trên cùng máy (hoặc cluster) với quyền truy cập HDFS/Spark.
- Cấu hình biến môi trường SPARK_CMD, PROJECT_DIR, TRAFFIC_INPUT/HDFS_OUTPUT nếu cần.
"""

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_DIR = os.environ.get("TRAFFIC_PROJECT_DIR", "/home/lntuandat/traffic-bigdata")
SPARK_CMD = os.environ.get("TRAFFIC_SPARK_CMD", "spark-submit")
INPUT_PATH = os.environ.get(
    "TRAFFIC_INPUT",
    "hdfs://localhost:9000/data/traffic/METR-LA.csv",
)
OUTPUT_PATH = os.environ.get(
    "TRAFFIC_OUTPUT",
    "hdfs://localhost:9000/results/metr_la",
)
PLOTS_LOCAL = os.environ.get("TRAFFIC_PLOTS_LOCAL", f"{PROJECT_DIR}/results")

default_args = {
    "owner": "traffic",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="traffic_bigdata_pipeline",
    default_args=default_args,
    schedule_interval="0 * * * *",  # mỗi giờ chạy một lần
    catchup=False,
    max_active_runs=1,
    tags=["traffic", "spark"],
) as dag:
    run_pipeline = BashOperator(
        task_id="run_spark_pipeline",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            f"{SPARK_CMD} --master local[*] main.py "
            f"--input {INPUT_PATH} "
            f"--out {OUTPUT_PATH} "
            f"--timecol timestamp "
            f"--has_header true "
            f"--plots_local {PLOTS_LOCAL}"
        ),
        env={
            "TRAFFIC_OUTPUT_DIR": PLOTS_LOCAL,
        },
    )

    sync_local = BashOperator(
        task_id="sync_results_local",
        bash_command=(
            f"cd {PROJECT_DIR} && "
            "mkdir -p results results/plots_kmeans results/silhouette_scores "
            "results/traffic_signal_plan results/traffic_signal_plan_json results/analysis results/models && "
            "hdfs dfs -get -f "
            f"{OUTPUT_PATH}/predict_* results/ && "
            f"hdfs dfs -get -f {OUTPUT_PATH}/plots_kmeans results/plots_kmeans && "
            f"hdfs dfs -get -f {OUTPUT_PATH}/silhouette_scores results/silhouette_scores && "
            f"hdfs dfs -get -f {OUTPUT_PATH}/traffic_signal_plan results/traffic_signal_plan && "
            f"hdfs dfs -get -f {OUTPUT_PATH}/traffic_signal_plan_json results/traffic_signal_plan_json && "
            f"hdfs dfs -get -f {OUTPUT_PATH}/analysis results/analysis && "
            f"hdfs dfs -get -f {OUTPUT_PATH}/models results/models"
        ),
    )

    run_pipeline >> sync_local
