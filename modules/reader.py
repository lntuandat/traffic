import pandas as pd
from functools import reduce
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.sql.window import Window

def read_and_clean(spark, args):
    df = (
        spark.read.option("header", args.has_header.lower() == "true")
        .option("inferSchema", True)
        .option("delimiter", args.delimiter)
        .csv(args.input)
    )

    # 🔧 Xác định cột timestamp
    if args.timecol not in df.columns:
        time_col = df.columns[0]
        print(f"⚠️ Không thấy '{args.timecol}', dùng '{time_col}'")
        df = df.withColumnRenamed(time_col, "timestamp")
    else:
        df = df.withColumnRenamed(args.timecol, "timestamp")

    # ✅ Ép định dạng timestamp kiểu Mỹ M/d/yyyy H:mm
    df = df.withColumn("timestamp", F.to_timestamp(F.col("timestamp").cast(StringType()), "M/d/yyyy H:mm"))

    print("📊 Kiểm tra timestamp sau khi ép kiểu:")
    df.select("timestamp").show(5, truncate=False)
    valid_ts = df.filter(F.col("timestamp").isNotNull()).count()
    print(f"✅ Timestamp hợp lệ: {valid_ts:,}")

    if valid_ts == 0:
        print("❌ Không có timestamp hợp lệ — kiểm tra định dạng CSV.")
        return df.limit(0)

    sensor_cols = [c for c in df.columns if c != "timestamp"]

    exprs = [
        F.when((F.col(c) <= 0) | (F.col(c) > 120), None)
         .otherwise(F.col(c).cast("double"))
         .alias(c)
        for c in sensor_cols
    ]
    df = df.select("timestamp", *exprs)
    print("🧹 Làm sạch dữ liệu thành công (đã loại bỏ giá trị lỗi & ép kiểu double).")
    df.show(3)

    # 🔍 Thống kê giá trị thiếu theo yêu cầu
    missing_stats = (
        df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
          .toPandas()
          .T
          .reset_index()
    )
    missing_stats.columns = ["Cột", "Số_giá_trị_thiếu"]
    total_rows = df.count()
    if total_rows == 0:
        print("⚠️ DataFrame trống sau làm sạch.")
        return df

    missing_stats["Tỉ_lệ_%"] = (missing_stats["Số_giá_trị_thiếu"] / total_rows * 100).round(2)

    print(f"📊 Tổng số dòng: {total_rows:,}")
    print("🔎 Top 10 cột có nhiều giá trị thiếu nhất:")
    print(missing_stats.sort_values("Số_giá_trị_thiếu", ascending=False).head(10).to_string(index=False))

    # 🧹 Xử lý giá trị thiếu bằng trung bình từng cột
    print("🔎 Đang tính giá trị trung bình cho các cột...")
    mean_dict = {}
    for c in df.columns:
        if c != "timestamp":
            mean_val = df.select(F.mean(F.col(c))).collect()[0][0]
            if mean_val is not None:
                mean_dict[c] = float(mean_val)

    print(f"✅ Đã tính trung bình cho {len(mean_dict)} cột có dữ liệu hợp lệ.")
    df = df.fillna(mean_dict)

    null_count = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns])
    print("📋 Số giá trị thiếu sau khi fillna:")
    null_count.show(truncate=False)
    print("✅ Đã xử lý xong toàn bộ giá trị thiếu.")

    # ⚙️ Tính tốc độ trung bình & đặc trưng thời gian
    sensor_cols = [c for c in df.columns if c != "timestamp"]
    if not sensor_cols:
        print("⚠️ Không có cột cảm biến nào để tính avg_speed.")
    else:
        col_exprs = [F.col(c) for c in sensor_cols]
        sum_expr = col_exprs[0] if len(col_exprs) == 1 else reduce(lambda a, b: a + b, col_exprs)
        df = df.withColumn("avg_speed", sum_expr / len(sensor_cols))

    df = df.withColumn("timestamp", F.to_timestamp("timestamp"))
    df = df.withColumn("hour", F.hour("timestamp"))
    df = df.withColumn("weekday", F.dayofweek("timestamp"))

    window = Window.partitionBy("hour")
    df = df.withColumn("std_speed_hour", F.stddev_samp("avg_speed").over(window))
    df = df.fillna({"std_speed_hour": 0.0})
    df = df.filter(F.col("avg_speed").isNotNull())

    if df.rdd.isEmpty():
        print("⚠️ Không còn dữ liệu sau khi tính avg_speed.")
        return df

    print("✅ Đã tính avg_speed, hour, weekday, std_speed_hour thành công.")
    df.select("timestamp", "avg_speed", "hour", "weekday", "std_speed_hour").show(10, truncate=False)

    # 💥 Loại bỏ ngoại lai cực mạnh (Hard++)
    stats = df.select(
        F.expr("percentile_approx(avg_speed, 0.25)").alias("Q1"),
        F.expr("percentile_approx(avg_speed, 0.75)").alias("Q3")
    ).collect()[0]
    Q1, Q3 = stats.Q1, stats.Q3
    IQR = Q3 - Q1 if Q1 is not None and Q3 is not None else None

    if IQR is None:
        print("⚠️ Không thể tính IQR cho avg_speed.")
    else:
        lower_bound = Q1 - 2.5 * IQR
        upper_bound = Q3 + 2.5 * IQR

        count_before = df.count()
        df_strict = df.filter(
            (F.col("avg_speed") >= lower_bound) &
            (F.col("avg_speed") <= upper_bound) &
            (F.col("avg_speed") > 5) &
            (F.col("avg_speed") < 110)
        )
        count_after = df_strict.count()
        removed = count_before - count_after

        print(f"📊 Tổng số bản ghi ban đầu: {count_before:,}")
        if count_before > 0:
            print(f"💣 Đã loại bỏ mạnh (Hard++): {removed:,} bản ghi ({removed / count_before * 100:.2f}%)")
        print(f"✅ Còn lại: {count_after:,} bản ghi hợp lệ")

        df_strict.select(
            F.mean("avg_speed").alias("Trung_bình"),
            F.min("avg_speed").alias("Tối_thiểu"),
            F.max("avg_speed").alias("Tối_đa")
        ).show()

        df = df_strict

    total = df.count()
    print(f"🧼 Sau làm sạch: còn {total:,} dòng")

    cleaned_out_path = f"{args.out}/cleaned_data"
    (
        df.orderBy("timestamp")
          .coalesce(1)
          .write.mode("overwrite").csv(cleaned_out_path, header=True)
    )
    print(f"💾 Đã lưu dữ liệu đã tiền xử lý tại {cleaned_out_path}")
    return df
