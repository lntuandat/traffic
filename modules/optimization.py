from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

def optimize_signals(pred_df, out_base, horizon_min):
    """
    Sinh gợi ý điều chỉnh chu kỳ đèn dựa trên xác suất tắc nghẽn.
    - Xác suất cao ⇒ kéo dài pha xanh cho hướng đang đông.
    - Xác suất thấp ⇒ có thể rút ngắn để ưu tiên hướng khác.
    Kết quả lưu ở {out_base}/traffic_signal_plan (JSON + CSV).
    """
    if pred_df is None or pred_df.rdd.isEmpty():
        print("⚠️ Không có dữ liệu dự báo để tối ưu đèn giao thông.")
        return None

    required = {"probability", "prediction_congested", "avg_speed", "timestamp"}
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        print(f"⚠️ Thiếu cột {missing} nên chưa thể tạo gợi ý đèn.")
        return None

    df = (
        pred_df
        .withColumn("prob_congestion", vector_to_array("probability").getItem(1))
        .withColumn("date", F.to_date("timestamp"))
    )

    agg = (
        df.groupBy("date", "hour")
          .agg(
              F.avg("prob_congestion").alias("avg_prob"),
              F.avg("avg_speed").alias("avg_speed"),
              F.count("*").alias("count_samples")
          )
    )

    plan = (
        agg.withColumn(
            "green_extension_pct",
            F.when(F.col("avg_prob") >= 0.7, F.lit(30))
             .when(F.col("avg_prob") >= 0.5, F.lit(20))
             .when(F.col("avg_prob") >= 0.3, F.lit(10))
             .otherwise(F.lit(-10))
        )
        .withColumn(
            "strategy",
            F.when(F.col("green_extension_pct") > 0,
                   F.concat(F.lit("Kéo dài pha xanh thêm "), F.col("green_extension_pct"), F.lit("%")))
             .otherwise(F.concat(F.lit("Rút ngắn pha xanh "), F.abs(F.col("green_extension_pct")), F.lit("%")))
        )
        .withColumn("horizon_min", F.lit(horizon_min))
        .orderBy("date", "hour")
    )

    plan.write.mode("overwrite").json(f"{out_base}/traffic_signal_plan_json")
    plan.write.mode("overwrite").csv(f"{out_base}/traffic_signal_plan", header=True)

    sample = plan.limit(5)
    print("🚦 Gợi ý điều chỉnh đèn giao thông (mẫu):")
    sample.show(truncate=False)
    return plan
