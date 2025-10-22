from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def run_prediction(df_with_clusters, out_base, horizon_min=15, freq_min=5):
    cleanup_cols = [c for c in ["features", "features_raw", "probability", "prediction_lr", "probability_lr"] if c in df_with_clusters.columns]
    if cleanup_cols:
        df_with_clusters = df_with_clusters.drop(*cleanup_cols)

    cluster_speed = (df_with_clusters.groupBy("prediction")
                     .agg(F.mean("avg_speed").alias("mean_v"))
                     .orderBy("mean_v")
                     .collect())

    if not cluster_speed:
        print("⚠️ Không tìm thấy cụm nào để dự báo.")
        return df_with_clusters.limit(0)

    congested = cluster_speed[0]["prediction"]
    lead_steps = max(1, int(horizon_min // max(1, freq_min)))

    w = Window.partitionBy(F.to_date("timestamp")).orderBy("timestamp")
    data_next = (
        df_with_clusters
        .withColumn("prediction_next", F.lead("prediction", lead_steps).over(w))
        .dropna()
    )
    data_bin = data_next.withColumn("label", F.when(F.col("prediction_next") == congested, 1).otherwise(0))
    data_bin = data_bin.cache()
    total_bin = data_bin.count()
    print(f"   • Tổng mẫu có nhãn (lead={lead_steps}): {total_bin:,}")

    label_stats = data_bin.groupBy("label").count().collect()
    if len(label_stats) < 2:
        print("⚠️ Chỉ có một lớp trong dữ liệu — không thể huấn luyện Logistic Regression.")
        data_bin.unpersist()
        return None

    counts = {row["label"]: row["count"] for row in label_stats}
    n_pos = counts.get(1, 0)
    n_neg = counts.get(0, 0)

    if n_pos == 0 or n_neg == 0:
        print("⚠️ Thiếu dữ liệu cho một trong hai lớp sau khi tạo nhãn.")
        data_bin.unpersist()
        return None

    pos_weight = float(n_neg) / float(n_pos)
    data_w = data_bin.withColumn(
        "weight",
        F.when(F.col("label") == 1, F.lit(pos_weight)).otherwise(F.lit(1.0))
    )
    data_w = data_w.cache()
    print(f"   • Trọng số lớp: pos_weight={pos_weight:.2f} (pos={n_pos:,}, neg={n_neg:,})")

    assembler = VectorAssembler(inputCols=["avg_speed", "hour", "weekday", "std_speed_hour"], outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="weight",
        maxIter=100,
        family="binomial",
        predictionCol="prediction_lr",
        probabilityCol="probability_lr"
    )

    pipe = Pipeline(stages=[assembler, scaler, lr])

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.0, 0.01, 0.05])
        .addGrid(lr.elasticNetParam, [0.0, 0.3])
        .build()
    )

    evaluator_pr = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    train_df, test_df = data_w.randomSplit([0.8, 0.2], seed=42)
    train_df = train_df.cache()
    test_df = test_df.cache()
    train_count = train_df.count()
    test_count = test_df.count()
    print(f"   • Chia train/test: {train_count:,} / {test_count:,}")

    train_df = train_df.withColumn("is_test", F.lit(False))
    test_df = test_df.withColumn("is_test", F.lit(True))
    combined_df = train_df.unionByName(test_df)

    if train_count < 10 or test_count < 5:
        print("⚠️ Dữ liệu train/test quá ít cho cross-validation. Bỏ qua bước dự báo.")
        train_df.unpersist()
        test_df.unpersist()
        data_w.unpersist()
        data_bin.unpersist()
        return None

    cv = CrossValidator(
        estimator=pipe,
        estimatorParamMaps=param_grid,
        evaluator=evaluator_pr,
        numFolds=2,
        seed=42
    )

    print(f"   • Đang chạy CrossValidator với {len(param_grid)} tổ hợp tham số × {cv.getNumFolds()} folds...")
    cv_model = cv.fit(train_df)
    pred_test = cv_model.transform(test_df)
    pred_full = cv_model.transform(combined_df)

    evaluator_roc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction_lr",
        metricName="accuracy"
    )
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction_lr",
        metricName="f1"
    )

    auc = evaluator_roc.evaluate(pred_test)
    auprc = evaluator_pr.evaluate(pred_test)
    acc = evaluator_acc.evaluate(pred_test)
    f1 = evaluator_f1.evaluate(pred_test)

    print(f"📈 AUC-ROC  : {auc:.4f}")
    print(f"📈 AUPRC    : {auprc:.4f}")
    print(f"🎯 Accuracy : {acc:.4f}")
    print(f"💡 F1-score : {f1:.4f}")

    pred_test.groupBy("label", "prediction_lr").count().orderBy("label", "prediction_lr").show()

    pred_full = (
        pred_full
        .withColumn("probability", F.col("probability_lr"))
        .withColumnRenamed("prediction_lr", "prediction_congested")
    )

    # Lưu hệ số đặc trưng để hiển thị độ quan trọng
    try:
        lr_stage = cv_model.bestModel.stages[-1]
        coeffs = [float(v) for v in lr_stage.coefficients]
        feature_names = ["avg_speed", "hour", "weekday", "std_speed_hour"]
        abs_vals = [abs(v) for v in coeffs]
        total_abs = sum(abs_vals) or 1.0
        rows = [
            (feature_names[i], coeffs[i], abs_vals[i] / total_abs)
            for i in range(len(feature_names))
        ]
        spark = df_with_clusters.sparkSession
        feature_df = spark.createDataFrame(rows, ["feature", "coefficient", "importance"])
        feature_df.orderBy(F.desc("importance")).write.mode("overwrite").json(
            f"{out_base}/analysis/logistic_feature_importance"
        )
    except Exception as err:
        print(f"⚠️ Không thể lưu độ quan trọng đặc trưng: {err}")

    train_df.unpersist()
    test_df.unpersist()
    data_w.unpersist()
    data_bin.unpersist()

    model_path = f"{out_base}/models/logistic_pipeline"
    try:
        cv_model.bestModel.write().overwrite().save(model_path)
        print(f"💾 Đã lưu mô hình LogisticRegression tại {model_path}")
    except Exception as err:
        print(f"⚠️ Không thể lưu mô hình logistic: {err}")

    pred_full.write.mode("overwrite").parquet(f"{out_base}/predict_{horizon_min}m")
    return pred_full
