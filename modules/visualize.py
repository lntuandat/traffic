import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np
from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

def plot_local(pred_df, out_dir, horizon_min):
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    pdf = (
        pred_df
        .orderBy("timestamp")
        .withColumn("prob_congestion", vector_to_array("probability").getItem(1))
        .select("timestamp", "prob_congestion")
        .limit(400)
        .toPandas()
    )
    plt.figure(figsize=(12,4))
    plt.plot(pdf["timestamp"], pdf["prob_congestion"])
    plt.title(f"Xác suất tắc nghẽn {horizon_min} phút tới")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/prob_congestion_{horizon_min}m.png", dpi=120)
    print(f"🖼  Đã lưu biểu đồ tại {out_dir}")

def plot_eval_metrics(pred_df, plots_dir, horizon_min, auc_roc=None):
    """
    Vẽ đường cong ROC và ma trận nhầm lẫn từ DataFrame dự báo (thường là tập test).
    pred_df: DataFrame có cột probability_lr (Vector) và prediction_lr, label.
    plots_dir: thư mục local để lưu hình (thường là args.plots_local).
    """
    if pred_df is None or not plots_dir:
        return

    metrics_dir = os.path.join(plots_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    score_df = (
        pred_df
        .select(
            vector_to_array("probability_lr").getItem(1).alias("prob_congestion"),
            F.col("label").cast("double").alias("label"),
            F.col("prediction_lr").cast("double").alias("prediction_lr")
        )
    )

    if score_df.rdd.isEmpty():
        print("⚠️ Không đủ dữ liệu để vẽ ROC/ma trận nhầm lẫn.")
        return

    pdf_scores = score_df.toPandas()
    if pdf_scores.empty:
        print("⚠️ Không thể tạo ROC vì tập test rỗng.")
        return

    pdf_scores = pdf_scores.sort_values("prob_congestion", ascending=False)
    labels = pdf_scores["label"].to_numpy()
    scores = pdf_scores["prob_congestion"].to_numpy()
    pos_total = max(1, int((labels == 1).sum()))
    neg_total = max(1, int((labels == 0).sum()))

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)
    tpr = np.concatenate(([0.0], tps / pos_total))
    fpr = np.concatenate(([0.0], fps / neg_total))
    auc_value = auc_roc if auc_roc is not None else float(np.trapz(tpr, fpr))

    # Chuẩn bị ma trận nhầm lẫn
    counts = score_df.groupBy("label", "prediction_lr").count().collect()
    cm = np.zeros((2, 2), dtype=int)
    for row in counts:
        i = int(row["label"])
        j = int(row["prediction_lr"])
        if 0 <= i < 2 and 0 <= j < 2:
            cm[i, j] = row["count"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ROC curve
    axes[0].plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC Curve ({horizon_min} phút)")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # Confusion matrix
    im = axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1].figure.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Không tắc", "Tắc"],
        yticklabels=["Không tắc", "Tắc"],
        ylabel="Thực tế",
        xlabel="Dự đoán"
    )
    axes[1].set_title("Ma trận nhầm lẫn")

    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(2):
        for j in range(2):
            axes[1].text(
                j, i, f"{cm[i, j]}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    out_path = os.path.join(metrics_dir, f"roc_confusion_{horizon_min}m.png")
    plt.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"🖼  Đã vẽ ROC & ma trận nhầm lẫn tại {out_path}")
