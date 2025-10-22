import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
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
