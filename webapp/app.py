import math
import os
from pathlib import Path

import json
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from typing import Optional
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel

DEFAULT_OUTPUT = "./plots"

app = Flask(__name__)
OUTPUT_DIR = Path(os.environ.get("TRAFFIC_OUTPUT_DIR", DEFAULT_OUTPUT)).resolve()
BASE_DIR = Path(__file__).resolve().parents[1]
METRICS_JSON = Path(os.environ.get("TRAFFIC_METRICS_JSON", BASE_DIR / "metrics" / "pipeline_metrics.json"))
MODEL_DIR = Path(os.environ.get("TRAFFIC_MODEL_DIR", OUTPUT_DIR / "models" / "logistic_pipeline"))

spark_session = None
logistic_model = None


def _coerce_float(value, allow_auto: bool = False) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if allow_auto and stripped.lower() == "auto":
            return None
        value = stripped
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _coerce_int(value, min_value: Optional[int] = None, max_value: Optional[int] = None) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        value = stripped
    try:
        intval = int(float(value))
    except (TypeError, ValueError):
        return None
    if min_value is not None and intval < min_value:
        return None
    if max_value is not None and intval > max_value:
        return None
    return intval

def _parse_timestamp(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (ValueError, OSError, OverflowError):
            return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            ts = pd.to_datetime(stripped)
        except Exception:
            return None
        if pd.isna(ts):
            return None
        return ts.to_pydatetime()
    return None


def list_prediction_runs():
    if not OUTPUT_DIR.exists():
        return []
    return sorted(
        [
            path.name
            for path in OUTPUT_DIR.iterdir()
            if path.is_dir() and path.name.startswith("predict_")
        ]
    )

def _infer_horizon_token(run_name: str) -> Optional[str]:
    if not run_name:
        return None
    parts = run_name.split("_")
    if not parts:
        return None
    suffix = parts[-1]
    suffix = suffix.strip().lower()
    if not suffix:
        return None
    if suffix.endswith(("m", "h")):
        core = suffix[:-1]
        if core.isdigit():
            return f"{int(core)}{suffix[-1]}"
        return suffix
    if suffix.isdigit():
        return f"{int(suffix)}m"
    return None

def find_evaluation_assets(run_name: str):
    metrics_dir = OUTPUT_DIR / "metrics"
    if not metrics_dir.exists():
        return []

    horizon = _infer_horizon_token(run_name)
    candidates = []
    if horizon:
        candidates.append(f"roc_confusion_{horizon}.png")
    candidates.append("roc_confusion.png")

    assets = []
    for name in candidates:
        candidate_path = metrics_dir / name
        if candidate_path.exists():
            assets.append(
                {
                    "title": "Đường cong ROC & Ma trận nhầm lẫn",
                    "filename": name,
                }
            )
    return assets


def read_prediction(run_name, limit=500):
    run_dir = OUTPUT_DIR / run_name
    if not run_dir.exists():
        abort(404, description=f"Không tìm thấy thư mục {run_name}")

    try:
        df = pd.read_parquet(run_dir)
    except Exception as exc:
        abort(500, description=f"Đọc parquet thất bại: {exc}")

    if "timestamp" not in df.columns:
        if {"date", "hour"}.issubset(df.columns):
            try:
                df["timestamp"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"].astype(int), unit="h")
            except Exception:
                abort(
                    500,
                    description=(
                        f"Dữ liệu trong {run_name} thiếu cột 'timestamp' "
                        "và không thể tái tạo từ 'date' và 'hour'. "
                        "Hãy chạy lại pipeline hoặc kiểm tra file parquet."
                    ),
                )
        else:
            abort(
                500,
                description=(
                    f"Dữ liệu trong {run_name} không chứa cột 'timestamp'. "
                    "Hãy chạy lại pipeline hoặc kiểm tra file parquet."
                ),
            )

    df = df.sort_values("timestamp")

    if "probability" in df.columns:
        df["prob_congestion"] = df["probability"].apply(_extract_prob)
    elif "probability_lr" in df.columns:
        df["prob_congestion"] = df["probability_lr"].apply(_extract_prob)
    else:
        df["prob_congestion"] = np.nan

    summary = {
        "total_rows": int(len(df)),
        "positive_cases": int(
            df[df.get("prediction_congested", df.get("prediction", 0)) == 1].shape[0]
        )
        if {"prediction_congested", "label"}.issubset(df.columns)
        else None,
    }

    metrics = _compute_metrics(df)

    display_cols = [
        c
        for c in df.columns
        if c
        not in {
            "features",
            "features_raw",
            "probability_lr",
            "rawPrediction",
            "weight",
        }
    ]
    sample = df[display_cols].tail(limit)
    chart_df = (
        df[["timestamp", "prob_congestion"]]
        .dropna()
        .tail(200)
    )
    chart_records = []
    for _, row in chart_df.iterrows():
        ts = row["timestamp"]
        if isinstance(ts, pd.Timestamp):
            ts = ts.strftime("%Y-%m-%d %H:%M")
        chart_records.append({"timestamp": ts, "prob": float(row["prob_congestion"])})

    return summary, metrics, sample, chart_records


def _extract_prob(value):
    if value is None:
        return np.nan
    if isinstance(value, dict):
        values = value.get("values")
        if isinstance(values, (list, tuple)) and len(values) > 1:
            return float(values[1])
        if isinstance(values, np.ndarray) and values.size > 1:
            return float(values[1])
        return np.nan
    if isinstance(value, (list, tuple)) and len(value) > 1:
        return float(value[1])
    if hasattr(value, "toArray"):
        arr = value.toArray()
        return float(arr[1]) if len(arr) > 1 else np.nan
    if hasattr(value, "__iter__"):
        as_list = list(value)
        return float(as_list[1]) if len(as_list) > 1 else np.nan
    if isinstance(value, str):
        return np.nan
    return float(value) if isinstance(value, (int, float)) else np.nan


def get_spark():
    global spark_session
    if spark_session is None:
        spark_session = (
            SparkSession.builder
            .master(os.environ.get("SPARK_MASTER", "local[*]"))
            .appName("TrafficRealtime")
            .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
            .config("spark.driver.bindAddress", os.environ.get("SPARK_DRIVER_BIND_ADDRESS", "127.0.0.1"))
            .config("spark.driver.host", os.environ.get("SPARK_DRIVER_HOST", "127.0.0.1"))
            .config("spark.ui.enabled", "false")
            .config("spark.socket.keepalive.timeout", "120s")
            .getOrCreate()
        )
        spark_session.sparkContext.setLogLevel("WARN")
    return spark_session


def get_logistic_model():
    global logistic_model
    if logistic_model is not None:
        return logistic_model
    if not MODEL_DIR.exists():
        return None
    spark = get_spark()
    logistic_model = PipelineModel.load(str(MODEL_DIR))
    return logistic_model


def _compute_metrics(df: pd.DataFrame):
    required_cols = {"label", "prediction_congested"}
    columns_present = {col for col in required_cols if col in df.columns}
    if columns_present != required_cols:
        return {}

    is_test_col = "is_test" in df.columns
    if is_test_col:
        data_eval = df[df["is_test"] == True]
        if data_eval.empty:
            data_eval = df
    else:
        data_eval = df

    try:
        y_true = data_eval["label"].astype(int)
        y_pred = data_eval["prediction_congested"].astype(int)
    except KeyError:
        return {}

    prob = data_eval.get("prob_congestion")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if prob is not None and prob.notna().sum() > 0:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob))
        except ValueError:
            metrics["roc_auc"] = None
        try:
            metrics["auprc"] = float(average_precision_score(y_true, prob))
        except ValueError:
            metrics["auprc"] = None

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    metrics["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    return metrics


def list_kmeans_plots():
    plots_dir = OUTPUT_DIR / "plots_kmeans"
    if not plots_dir.exists():
        return []
    return sorted({path.name for path in plots_dir.glob("*.png")})


def _sanitize_record(record):
    sanitized = {}
    for key, value in record.items():
        if isinstance(value, pd.Timestamp):
            sanitized[key] = value.strftime("%Y-%m-%d %H:%M")
        elif isinstance(value, np.generic):
            sanitized[key] = value.item()
        else:
            sanitized[key] = value
    return sanitized


def load_spark_json(dir_path):
    directory = Path(dir_path)
    if not directory.exists():
        return pd.DataFrame()
    files = sorted(directory.glob("*.json"))
    if not files:
        files = sorted(directory.glob("part-*"))
    if not files:
        return pd.DataFrame()
    return pd.read_json(files[0], lines=True)


def load_analysis():
    base = OUTPUT_DIR / "analysis"
    def dataset_records(name):
        candidates = [base / name, base / "analysis" / name]
        records = []
        for path in candidates:
            df = load_spark_json(path)
            if not df.empty:
                records.extend(_sanitize_record(r) for r in df.to_dict(orient="records"))
        return records

    overview_list = dataset_records("overview")
    overview = overview_list[0] if overview_list else {}

    def unique(records, key_fields):
        seen = set()
        result = []
        for rec in records:
            key = tuple(rec.get(k) for k in key_fields)
            if key not in seen:
                seen.add(key)
                result.append(rec)
        return result

    cluster = unique(dataset_records("cluster_distribution"), ["prediction"])
    cluster_total = 0
    for rec in cluster:
        if "prediction" in rec:
            rec["prediction"] = int(float(rec["prediction"]))
        if "avg_speed_mean" in rec:
            rec["avg_speed_mean"] = float(rec["avg_speed_mean"])
        elif "avg_speed" in rec:
            rec["avg_speed_mean"] = float(rec.pop("avg_speed"))
        if "row_count" in rec:
            rec["row_count"] = int(float(rec["row_count"]))
            cluster_total += rec["row_count"]
        elif "count" in rec:
            rec["row_count"] = int(float(rec.pop("count")))
            cluster_total += rec["row_count"]
        if "avg_prob_congestion" in rec:
            rec["avg_prob_congestion"] = float(rec["avg_prob_congestion"])

    top_congestion = unique(dataset_records("top_congestion"), ["date", "hour"])
    top_congestion = sorted(top_congestion, key=lambda r: (float(r.get("avg_prob", 0) or 0), float(r.get("count", 0) or 0)), reverse=True)
    for rec in top_congestion:
        if "hour" in rec and rec["hour"] is not None:
            rec["hour"] = int(float(rec["hour"]))
        if "count" in rec and rec["count"] is not None:
            rec["count"] = int(float(rec["count"]))
        if "avg_prob" in rec and rec["avg_prob"] is not None:
            rec["avg_prob"] = float(rec["avg_prob"])

    hourly_records = unique(dataset_records("hourly_summary"), ["hour"])
    std_speed_sum = 0.0
    std_speed_count = 0
    hourly = []
    for rec in hourly_records:
        if "hour" in rec and rec["hour"] is not None:
            rec["hour"] = int(float(rec["hour"]))
        if "avg_speed" in rec and rec["avg_speed"] is not None:
            rec["avg_speed"] = float(rec["avg_speed"])
        if "avg_prob_congestion" in rec and rec["avg_prob_congestion"] is not None:
            rec["avg_prob_congestion"] = float(rec["avg_prob_congestion"])
        if "count" in rec and rec["count"] is not None:
            rec["count"] = int(float(rec["count"]))
        if "std_speed_hour" in rec and rec["std_speed_hour"] is not None:
            rec["std_speed_hour"] = float(rec["std_speed_hour"])
            std_speed_sum += rec["std_speed_hour"]
            std_speed_count += 1
        hourly.append(rec)
    hourly.sort(key=lambda r: r.get("hour", 0))

    hourly_std_avg = (std_speed_sum / std_speed_count) if std_speed_count else None

    weekday_records = unique(dataset_records("weekday_summary"), ["weekday"])
    weekday = []
    for rec in weekday_records:
        if "weekday" in rec and rec["weekday"] is not None:
            rec["weekday"] = int(float(rec["weekday"]))
        if "avg_speed" in rec and rec["avg_speed"] is not None:
            rec["avg_speed"] = float(rec["avg_speed"])
        if "avg_prob_congestion" in rec and rec["avg_prob_congestion"] is not None:
            rec["avg_prob_congestion"] = float(rec["avg_prob_congestion"])
        if "count" in rec and rec["count"] is not None:
            rec["count"] = int(float(rec["count"]))
        weekday.append(rec)
    weekday.sort(key=lambda r: r.get("weekday", 0))

    if hourly_std_avg is None:
        hourly_std_avg = compute_hourly_std_from_predictions()

    feature_importance = dataset_records("logistic_feature_importance")
    for rec in feature_importance:
        if "coefficient" in rec and rec["coefficient"] is not None:
            rec["coefficient"] = float(rec["coefficient"])
        if "importance" in rec and rec["importance"] is not None:
            rec["importance"] = float(rec["importance"])
    feature_importance = sorted(feature_importance, key=lambda r: r.get("importance", 0), reverse=True)

    return {
        "overview": overview,
        "cluster": cluster,
        "cluster_total": cluster_total,
        "congestion": dataset_records("congestion_summary"),
        "hourly": hourly,
        "hourly_std_avg": hourly_std_avg,
        "weekday": weekday,
        "top": top_congestion[:20],
        "histogram": dataset_records("histogram_avg_speed"),
        "daily": dataset_records("daily_avg_speed"),
        "heatmap": dataset_records("weekday_hour_heatmap"),
        "sensor_activity": dataset_records("sensor_activity"),
        "top_sensor_hourly": dataset_records("top_sensor_hourly"),
        "feature_importance": feature_importance,
    }


def compute_hourly_std_from_predictions() -> Optional[float]:
    candidates = sorted(
        OUTPUT_DIR.glob("predict_*/*.parquet"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for file in candidates:
        try:
            df = pd.read_parquet(file)
        except Exception:
            continue
        if not {"hour", "avg_speed"}.issubset(df.columns):
            continue
        std_series = df.groupby("hour")["avg_speed"].std(ddof=1).dropna()
        if not std_series.empty:
            return float(std_series.mean())
    return None


def load_pipeline_metrics():
    if not METRICS_JSON.exists():
        return {}
    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    ts = data.get("timestamp")
    if ts:
        data["last_run_human"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    data.setdefault("durations", {})
    data.setdefault("metadata", {})
    return data


def load_signal_plan(limit: int = 50):
    plan_dir = OUTPUT_DIR / "traffic_signal_plan"
    result = {
        "records": [],
        "top": [],
        "last_updated": None,
        "last_updated_human": None,
    }
    if not plan_dir.exists():
        return result

    csv_files = sorted(
        plan_dir.glob("**/*.csv"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if not csv_files:
        return result

    frames = []
    latest_mtime = None
    for file in csv_files:
        try:
            df = pd.read_csv(file)
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
        if latest_mtime is None:
            try:
                latest_mtime = file.stat().st_mtime
            except OSError:
                latest_mtime = None
        if len(frames) >= 10:
            break

    if not frames:
        return result

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        return result

    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        df["date"] = df["date_str"].where(df["date_str"].notna(), df["date"].astype(str))
        df.drop(columns=["date_str"], inplace=True)
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    for field in ["avg_prob", "avg_speed"]:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")
    if "count_samples" in df.columns:
        df["count_samples"] = pd.to_numeric(df["count_samples"], errors="coerce")
    if "green_extension_pct" in df.columns:
        df["green_extension_pct"] = pd.to_numeric(df["green_extension_pct"], errors="coerce")
    if "horizon_min" in df.columns:
        df["horizon_min"] = pd.to_numeric(df["horizon_min"], errors="coerce")

    df = df.dropna(subset=["date"])
    df_sorted = df.sort_values(["date", "hour"], na_position="last")

    records = []
    for _, row in df_sorted.head(limit).iterrows():
        record = {
            "date": row.get("date"),
            "hour": int(row["hour"]) if pd.notna(row.get("hour")) else None,
            "avg_prob": float(row["avg_prob"]) if pd.notna(row.get("avg_prob")) else None,
            "avg_speed": float(row["avg_speed"]) if pd.notna(row.get("avg_speed")) else None,
            "count_samples": int(row["count_samples"]) if pd.notna(row.get("count_samples")) else None,
            "green_extension_pct": int(row["green_extension_pct"]) if pd.notna(row.get("green_extension_pct")) else None,
            "strategy": row.get("strategy"),
            "horizon_min": int(row["horizon_min"]) if pd.notna(row.get("horizon_min")) else None,
        }
        rec_sanitized = _sanitize_record(record)
        if rec_sanitized.get("date") and isinstance(rec_sanitized["date"], datetime):
            rec_sanitized["date"] = rec_sanitized["date"].strftime("%Y-%m-%d")
        if rec_sanitized.get("avg_prob") is not None:
            rec_sanitized["avg_prob_percent"] = round(rec_sanitized["avg_prob"] * 100, 1)
        if rec_sanitized.get("avg_speed") is not None:
            rec_sanitized["avg_speed"] = round(rec_sanitized["avg_speed"], 2)
        records.append(rec_sanitized)

    top_records = sorted(
        records,
        key=lambda r: (
            r.get("avg_prob") or 0.0,
            r.get("count_samples") or 0,
        ),
        reverse=True,
    )[:5]

    result["records"] = records
    result["top"] = top_records
    if latest_mtime:
        result["last_updated"] = latest_mtime
        result["last_updated_human"] = datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M")
    return result


def _signal_strategy(prob: Optional[float]) -> dict:
    if prob is None or (
        isinstance(prob, (float, np.floating)) and math.isnan(float(prob))
    ):
        return {
            "level": "unknown",
            "green_extension_pct": 0,
            "strategy": "Giữ nguyên chu kỳ đèn hiện tại",
        }
    if prob >= 0.7:
        return {
            "level": "critical",
            "green_extension_pct": 30,
            "strategy": "Kéo dài pha xanh thêm 30%",
        }
    if prob >= 0.5:
        return {
            "level": "high",
            "green_extension_pct": 20,
            "strategy": "Kéo dài pha xanh thêm 20%",
        }
    if prob >= 0.3:
        return {
            "level": "moderate",
            "green_extension_pct": 10,
            "strategy": "Kéo dài pha xanh thêm 10%",
        }
    return {
        "level": "low",
        "green_extension_pct": -10,
        "strategy": "Rút ngắn pha xanh 10% để ưu tiên hướng khác",
    }



def load_hourly_std_default(hour: int, default: float = 5.0) -> float:
    base = OUTPUT_DIR / "analysis" / "hourly_summary"
    df = load_spark_json(base)
    if df.empty:
        return default
    # nếu bảng chưa có cột std_speed_hour -> fallback
    if "std_speed_hour" not in df.columns:
        return float(default)
    match = df[df["hour"] == hour]
    if match.empty:
        return float(df["std_speed_hour"].mean() if "std_speed_hour" in df.columns else default)
    value = match.iloc[0]["std_speed_hour"]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@app.route("/")
def index():
    runs = sorted({run for run in list_prediction_runs()})
    plots = list_kmeans_plots()
    analysis = load_analysis()
    metrics = load_pipeline_metrics()
    signal_plan = load_signal_plan()
    return render_template(
        "index.html",
        output_dir=str(OUTPUT_DIR),
        runs=runs,
        plots=plots,
        analysis=analysis,
        metrics=metrics,
        signal_plan=signal_plan,
    )



@app.route("/prediction/<run_name>")
def prediction_detail(run_name):
    summary, metrics, sample, chart = read_prediction(run_name)
    columns = list(sample.columns)
    records = sample.to_dict(orient="records")
    eval_assets = find_evaluation_assets(run_name)
    return render_template(
        "prediction.html",
        run_name=run_name,
        summary=summary,
        metrics=metrics,
        columns=columns,
        records=records,
        chart=chart,
        eval_assets=eval_assets,
    )


@app.route("/plots/<path:filename>")
def serve_plot(filename):
    plots_dir = OUTPUT_DIR / "plots_kmeans"
    if not plots_dir.exists():
        abort(404)
    return send_from_directory(plots_dir, filename)

@app.route("/metrics/<path:filename>")
def serve_metric(filename):
    metrics_dir = OUTPUT_DIR / "metrics"
    if not metrics_dir.exists():
        abort(404)
    return send_from_directory(metrics_dir, filename)

@app.route("/api/realtime", methods=["POST"])
def api_realtime():
    if not request.is_json:
        return jsonify({"error": "Yêu cầu phải gửi dữ liệu JSON."}), 415

    payload = request.get_json(silent=True) or {}
    errors = []

    avg_speed = _coerce_float(payload.get("avg_speed"))
    if avg_speed is None:
        errors.append("Thiếu hoặc sai định dạng trường avg_speed.")

    timestamp_raw = payload.get("timestamp")
    timestamp_dt = _parse_timestamp(timestamp_raw)
    now = datetime.utcnow()

    hour = _coerce_int(payload.get("hour"), 0, 23)
    if hour is None and timestamp_dt is not None:
        hour = timestamp_dt.hour
    if hour is None:
        hour = now.hour

    weekday = _coerce_int(payload.get("weekday"), 1, 7)
    if weekday is None and timestamp_dt is not None:
        weekday = ((timestamp_dt.weekday() + 1) % 7) + 1
    if weekday is None:
        weekday = ((now.weekday() + 1) % 7) + 1

    std_speed_raw = payload.get("std_speed_hour")
    std_speed_auto = False
    std_speed_hour = None
    if isinstance(std_speed_raw, str) and std_speed_raw.strip().lower() == "auto":
        std_speed_hour = load_hourly_std_default(hour)
        std_speed_auto = True
    else:
        std_speed_hour = _coerce_float(std_speed_raw)
        if std_speed_hour is None:
            std_speed_hour = load_hourly_std_default(hour)
            std_speed_auto = True

    latitude = _coerce_float(payload.get("latitude"))
    longitude = _coerce_float(payload.get("longitude"))
    location = payload.get("location")

    if errors:
        return jsonify({"error": "Dữ liệu đầu vào chưa hợp lệ.", "details": errors}), 400

    model = get_logistic_model()
    if model is None:
        return jsonify({"error": "Chưa tìm thấy mô hình logistic. Hãy chạy pipeline trước."}), 503

    spark = get_spark()
    feature_row = {
        "avg_speed": float(avg_speed),
        "hour": int(hour),
        "weekday": int(weekday),
        "std_speed_hour": float(std_speed_hour),
    }

    try:
        df = spark.createDataFrame([feature_row])
        pred_row = (
            model.transform(df)
            .select("probability_lr", "prediction_lr", "rawPrediction")
            .first()
        )
    except Exception as exc:
        app.logger.exception("Realtime prediction failed: %s", exc)
        return jsonify({"error": "Không thể tính dự báo thời gian thực.", "details": str(exc)}), 500

    if pred_row is None:
        return jsonify({"error": "Không nhận được kết quả dự báo."}), 500

    prob_vector = pred_row["probability_lr"]
    prob_congestion = _extract_prob(prob_vector)

    strategy = _signal_strategy(prob_congestion)
    prediction_label = int(pred_row["prediction_lr"])
    status = "Có nguy cơ tắc nghẽn" if prediction_label == 1 else "Thông thoáng"

    probability_distribution = None
    if prob_vector is not None:
        if hasattr(prob_vector, "toArray"):
            probability_distribution = [float(v) for v in prob_vector.toArray()]
        elif isinstance(prob_vector, (list, tuple, np.ndarray)):
            probability_distribution = [float(v) for v in prob_vector]

    model_info = None
    if MODEL_DIR.exists():
        try:
            model_info = {
                "path": str(MODEL_DIR),
                "last_modified": datetime.fromtimestamp(MODEL_DIR.stat().st_mtime).isoformat()
            }
        except OSError:
            model_info = {"path": str(MODEL_DIR)}

    response = {
        "status": status,
        "prediction": prediction_label,
        "prob_congestion": float(prob_congestion) if prob_congestion is not None else None,
        "probability_distribution": probability_distribution,
        "strategy": strategy,
        "inputs": _sanitize_record({
            "location": location,
            "avg_speed": float(avg_speed) if avg_speed is not None else None,
            "hour": hour,
            "weekday": weekday,
            "std_speed_hour": float(std_speed_hour) if std_speed_hour is not None else None,
            "std_speed_auto": std_speed_auto,
            "timestamp": (timestamp_dt or now).isoformat(),
            "latitude": float(latitude) if latitude is not None else None,
            "longitude": float(longitude) if longitude is not None else None,
        }),
        "model": model_info,
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
