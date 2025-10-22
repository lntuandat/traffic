from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array


def analyze_traffic(pred_df, out_base, original_df=None):
    """
    Sinh thống kê tổng hợp để phục vụ dashboard:
    - Phân bố cụm (prediction)
    - Phân bố xác suất tắc nghẽn theo giờ, ngày
    - Tổng quan (tổng số dòng, tốc độ TB, tỷ lệ tắc nghẽn)
    """
    if pred_df is None or pred_df.rdd.isEmpty():
        print("⚠️ Không có dữ liệu dự báo để phân tích.")
        return

    required = {"probability", "prediction_congested", "avg_speed", "timestamp"}
    missing = [c for c in required if c not in pred_df.columns]
    if missing:
        print(f"⚠️ Thiếu cột {missing} nên bỏ qua bước phân tích.")
        return

    spark = pred_df.sparkSession

    df = (
        pred_df
        .withColumn("prob_congestion", vector_to_array("probability").getItem(1))
        .withColumn("date", F.to_date("timestamp"))
    )

    overview = (
        df.agg(
            F.count("*").alias("total_rows"),
            F.avg("avg_speed").alias("avg_speed"),
            F.avg("prob_congestion").alias("avg_prob_congestion"),
            F.avg(F.col("prediction_congested").cast("double")).alias("congestion_ratio"),
            F.min("timestamp").alias("start_time"),
            F.max("timestamp").alias("end_time"),
        )
        .withColumn("generated_at", F.current_timestamp())
    )
    overview.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/overview")

    cluster_dist = (
        df.groupBy("prediction")
          .agg(
              F.count("*").alias("count"),
              F.avg("avg_speed").alias("avg_speed"),
              F.avg("prob_congestion").alias("avg_prob_congestion")
          )
          .orderBy("prediction")
    )
    cluster_dist.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/cluster_distribution")

    congestion_summary = (
        df.groupBy("prediction_congested")
          .agg(
              F.count("*").alias("count"),
              F.avg("prob_congestion").alias("avg_prob"),
              F.avg("avg_speed").alias("avg_speed")
          )
          .orderBy("prediction_congested")
    )
    congestion_summary.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/congestion_summary")

    hourly = (
        df.groupBy("hour")
          .agg(
              F.avg("avg_speed").alias("avg_speed"),
              F.avg("prob_congestion").alias("avg_prob_congestion"),
              F.stddev_samp("avg_speed").alias("std_speed_hour"),
              F.count("*").alias("count")
          )
          .orderBy("hour")
    )
    hourly.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/hourly_summary")

    weekday = (
        df.groupBy("weekday")
          .agg(
              F.avg("avg_speed").alias("avg_speed"),
              F.avg("prob_congestion").alias("avg_prob_congestion"),
              F.stddev_samp("avg_speed").alias("std_speed_weekday"),
              F.count("*").alias("count")
          )
          .orderBy("weekday")
    )
    weekday.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/weekday_summary")

    top_congestion = (
        df.filter(F.col("prediction_congested") == 1)
          .groupBy("date", "hour")
          .agg(
              F.avg("prob_congestion").alias("avg_prob"),
              F.count("*").alias("count")
          )
          .orderBy(F.desc("avg_prob"), F.desc("count"))
          .limit(20)
    )
    top_congestion.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/top_congestion")

    # Histogram của avg_speed (mỗi 5 km/h)
    bin_size = 5.0
    histogram = (
        df.select(F.col("avg_speed"))
          .where(F.col("avg_speed").isNotNull())
          .withColumn("bin_lower", F.floor(F.col("avg_speed") / bin_size) * bin_size)
          .groupBy("bin_lower")
          .agg(F.count("*").alias("count"))
          .withColumn("bin_upper", F.col("bin_lower") + bin_size)
          .orderBy("bin_lower")
    )
    histogram.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/histogram_avg_speed")

    # Tốc độ trung bình theo ngày
    daily = (
        df.groupBy("date")
          .agg(
              F.avg("avg_speed").alias("avg_speed"),
              F.stddev_samp("avg_speed").alias("std_speed"),
              F.avg("prob_congestion").alias("avg_prob_congestion"),
              F.count("*").alias("count")
          )
          .orderBy("date")
    )
    daily.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/daily_avg_speed")

    # Heatmap giờ x ngày trong tuần
    weekday_hour = (
        df.groupBy("weekday", "hour")
          .agg(
              F.avg("avg_speed").alias("avg_speed"),
              F.avg("prob_congestion").alias("avg_prob_congestion"),
              F.count("*").alias("count")
          )
          .orderBy("weekday", "hour")
    )
    weekday_hour.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/weekday_hour_heatmap")

    # Hoạt động cảm biến (nếu DataFrame gốc được cung cấp)
    if original_df is not None:
        base_cols = {"timestamp", "avg_speed", "hour", "weekday", "std_speed_hour", "cluster_label", "prediction"}
        sensor_cols = [c for c in original_df.columns if c not in base_cols]
        if sensor_cols:
            activity_exprs = [F.count(F.when(F.col(c).isNotNull(), 1)).alias(c) for c in sensor_cols]
            activity_row = original_df.select(*activity_exprs).collect()[0].asDict()
            activity_data = [(name, int(activity_row[name])) for name in sensor_cols]
            sensor_activity = spark.createDataFrame(activity_data, ["sensor", "count"])
            sensor_activity.orderBy(F.desc("count")).coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/sensor_activity")

            top_sensors = [row["sensor"] for row in sensor_activity.orderBy(F.desc("count")).limit(5).collect()]
            if top_sensors:
                # Chuyển sang dạng long để tính trung bình theo giờ cho top cảm biến
                stacked_expr = ", ".join([f"'{s}', {s}" for s in top_sensors])
                long_df = (
                    original_df.select("timestamp", "hour", *top_sensors)
                                .select(
                                    "timestamp",
                                    "hour",
                                    F.expr(f"stack({len(top_sensors)}, {stacked_expr}) as (sensor, speed)")
                                )
                                .where(F.col("speed").isNotNull())
                )
                sensor_hourly = (
                    long_df.groupBy("sensor", "hour")
                           .agg(F.avg("speed").alias("avg_speed"))
                           .orderBy("sensor", "hour")
                )
                sensor_hourly.coalesce(1).write.mode("overwrite").json(f"{out_base}/analysis/top_sensor_hourly")

    print("📊 Đã sinh thống kê phân tích cho dashboard.")
