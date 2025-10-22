import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def run_kmeans(df, out_base, kmin, kmax):
    """
    Chạy thuật toán KMeans để phân cụm dữ liệu giao thông.

    Hàm này sẽ tự động tìm số cụm (k) tối ưu trong khoảng [kmin, kmax]
    dựa trên chỉ số Silhouette, sau đó gán nhãn và trực quan hóa kết quả.

    Args:
        df (DataFrame): DataFrame đầu vào chứa dữ liệu giao thông.
        out_base (str): Đường dẫn thư mục để lưu kết quả (model, summary, plots).
        kmin (int): Số cụm tối thiểu để thử nghiệm.
        kmax (int): Số cụm tối đa để thử nghiệm.

    Returns:
        tuple: (DataFrame chứa dự đoán với nhãn, k tối ưu được chọn).
    """
    print("🚀 Bắt đầu quy trình tối ưu và huấn luyện KMeans...")

    # 1. Làm sạch dữ liệu
    clean_df = (
        df.filter(F.col("avg_speed").isNotNull())
          .filter(~F.isnan(F.col("avg_speed")))
          .filter(~F.col("avg_speed").isin(float("inf"), float("-inf")))
    )

    total = df.count()
    clean_count = clean_df.count()
    removed = total - clean_count
    if removed:
        print(f"🧹 Loại bỏ {removed:,} dòng có giá trị 'avg_speed' không hợp lệ.")

    if clean_count == 0:
        print("❌ Không còn dữ liệu hợp lệ để chạy KMeans. Dừng quy trình.")
        return df.limit(0), None

    # 2. Chuẩn bị features
    assembler = VectorAssembler(
        inputCols=["avg_speed", "hour", "weekday", "std_speed_hour"],
        outputCol="features",
        handleInvalid="skip"
    )
    data = assembler.transform(clean_df).select("timestamp", "avg_speed", "hour", "weekday", "std_speed_hour", "features").cache()
    
    # 3. Vòng lặp tìm k tối ưu
    print(f"\n🔍 Bắt đầu tìm k tối ưu trong khoảng [{kmin}, {kmax}]...")
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette", distanceMeasure="squaredEuclidean")

    best_k = -1
    best_silhouette = -1.0  # Silhouette score nằm trong khoảng [-1, 1]
    best_preds = None
    silhouette_scores = []

    for k in range(kmin, kmax + 1):
        print(f"   - Đang huấn luyện với k={k}...")
        kmeans = KMeans(
            k=k,
            seed=42,
            featuresCol="features",
            maxIter=120,
            tol=1e-4,
            initMode="k-means||",
            initSteps=5
        )
        model = kmeans.fit(data)
        preds = model.transform(data)
        silhouette = evaluator.evaluate(preds)
        silhouette_scores.append((k, silhouette))
        print(f"     => Kết quả: k={k}, Silhouette Score = {silhouette:.4f}")

        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
            best_preds = preds

    print(f"\n✅ Đã tìm thấy k tối ưu: k={best_k} với Silhouette Score = {best_silhouette:.4f}")
    
    # Lưu và trực quan hóa điểm silhouette
    plots_dir = os.path.join(out_base, "plots_kmeans")
    os.makedirs(plots_dir, exist_ok=True)
    
    spark = df.sparkSession
    k_scores_df = spark.createDataFrame(silhouette_scores, ["k", "silhouette"])
    k_scores_df.coalesce(1).write.mode("overwrite").csv(f"{out_base}/silhouette_scores", header=True)
    
    k_scores_pd = k_scores_df.toPandas()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=k_scores_pd, x="k", y="silhouette", marker='o', color='royalblue')
    plt.title("📈 Đánh giá Silhouette cho các giá trị K")
    plt.xlabel("Số cụm (k)")
    plt.ylabel("Silhouette Score")
    plt.xticks(range(kmin, kmax + 1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "silhouette_scores.png"), dpi=120)
    plt.close()
    print(f"🎨 Đã lưu biểu đồ đánh giá Silhouette vào {plots_dir}")

    # 4. Phân tích và gán nhãn cho mô hình tốt nhất
    print("\n📊 Phân tích kết quả từ mô hình tối ưu...")
    
    summary = (
        best_preds.groupBy("prediction")
                  .agg(
                      F.mean("avg_speed").alias("avg_speed_mean"),
                      F.count("*").alias("row_count")
                  )
                  .orderBy("avg_speed_mean")
    )

    rank_window = Window.orderBy("avg_speed_mean")
    summary = (
        summary
        .withColumn("rank_by_speed", F.row_number().over(rank_window))
        .withColumn(
            "cluster_label",
            F.when(F.col("rank_by_speed") == 1, F.lit("Tắc nghẽn"))
             .when(F.col("rank_by_speed") == best_k, F.lit("Thông thoáng"))
             .otherwise(F.lit("Bình thường"))
        )
        .orderBy("prediction")
    )

    print("\nKết quả gom cụm (đã gán nhãn):")
    summary.select("prediction", "avg_speed_mean", "row_count", "cluster_label").show(truncate=False)
    summary.coalesce(1).write.mode("overwrite").csv(f"{out_base}/cluster_summary", header=True)

    # Gắn nhãn vào DataFrame dự báo
    final_preds = best_preds.join(summary.select("prediction", "cluster_label"), on="prediction", how="left")

    # 5. Trực quan hóa kết quả cuối cùng
    print("\n🎨 Bắt đầu trực quan hóa kết quả phân cụm...")
    pdf = (
        final_preds.select("timestamp", "avg_speed", "prediction", "cluster_label")
                   .orderBy("timestamp")
                   .limit(5000)
                   .toPandas()
    )
    if not pdf.empty:
        pdf = pdf.sort_values("timestamp")
        
        plt.figure(figsize=(14, 7))
        sns.scatterplot(data=pdf, x="timestamp", y="avg_speed", hue="cluster_label", s=15, alpha=0.9, palette="viridis")
        plt.title(f"🚦 Phân cụm tốc độ giao thông (k={best_k})")
        plt.xlabel("Thời gian")
        plt.ylabel("Tốc độ trung bình (km/h)")
        plt.xticks(rotation=45)
        plt.legend(title="Trạng thái giao thông", loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "scatter.png"), dpi=120)
        plt.close()

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=pdf, x="cluster_label", y="avg_speed", palette="viridis", order=["Tắc nghẽn", "Bình thường", "Thông thoáng"])
        plt.title("📦 Phân bố tốc độ theo trạng thái giao thông")
        plt.xlabel("Trạng thái giao thông")
        plt.ylabel("Tốc độ trung bình (km/h)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "boxplot.png"), dpi=120)
        plt.close()

        plt.figure(figsize=(10, 6))
        for label in ["Tắc nghẽn", "Bình thường", "Thông thoáng"]:
             if label in pdf['cluster_label'].unique():
                sns.kdeplot(
                    data=pdf[pdf["cluster_label"] == label],
                    x="avg_speed",
                    label=label,
                    fill=True,
                    alpha=0.5
                )
        plt.title("📈 Mật độ phân bố tốc độ theo trạng thái")
        plt.xlabel("Tốc độ trung bình (km/h)")
        plt.ylabel("Mật độ")
        plt.legend(title="Trạng thái")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "kde.png"), dpi=120)
        plt.close()

        print(f"✅ Đã lưu các biểu đồ phân tích vào thư mục: {plots_dir}")
    else:
        print("⚠️ Không đủ dữ liệu để vẽ biểu đồ KMeans.")

    data.unpersist()
    print("\n🎉 Hoàn tất quy trình KMeans.")
    return final_preds, best_k
