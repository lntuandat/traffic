import time

from modules.utils import build_argparser, start_spark, write_pipeline_metrics
from modules.reader import read_and_clean
from modules.clustering import run_kmeans
from modules.prediction import run_prediction
from modules.visualize import plot_local
from modules.optimization import optimize_signals
from modules.analytics import analyze_traffic

def main():
    args = build_argparser().parse_args()
    spark = start_spark()
    print("✅ Spark started")

    durations = {}
    t0 = time.time()

    print("⏳ (1/5) Đang đọc & làm sạch dữ liệu đầu vào...")
    t = time.time()
    df = read_and_clean(spark, args)
    durations["read_and_clean"] = time.time() - t
    print(f"✅ Hoàn tất bước 1 sau {durations['read_and_clean']:.1f}s\n")

    print("⏳ (2/5) Đang gom cụm KMeans để gắn nhãn hành vi...")
    t = time.time()
    preds, best_k = run_kmeans(df, args.out, args.kmin, args.kmax)
    durations["kmeans"] = time.time() - t
    print(f"✅ Hoàn tất bước 2 sau {durations['kmeans']:.1f}s\n")

    print("⏳ (3/5) Huấn luyện Logistic Regression dự báo tắc nghẽn...")
    t = time.time()
    pred_next = run_prediction(preds, args.out, args.horizon_min, args.freq_min, args.plots_local)
    durations["prediction"] = time.time() - t
    print(f"✅ Hoàn tất bước 3 sau {durations['prediction']:.1f}s\n")

    if pred_next is not None:
        print("⏳ (4/5) Vẽ biểu đồ xác suất tắc nghẽn...")
        t = time.time()
        plot_local(pred_next, args.plots_local, args.horizon_min)
        durations["plot_local"] = time.time() - t
        print(f"✅ Hoàn tất bước 4 sau {durations['plot_local']:.1f}s\n")

        print("⏳ (5/5) Tạo gợi ý đèn và thống kê phân tích...")
        t = time.time()
        optimize_signals(pred_next, args.out, args.horizon_min)
        durations["optimize_signals"] = time.time() - t

        t = time.time()
        analyze_traffic(pred_next, args.out, df)
        durations["analytics"] = time.time() - t
        print(f"✅ Hoàn tất bước 5 sau {durations['optimize_signals'] + durations['analytics']:.1f}s\n")
    else:
        durations["plot_local"] = 0.0
        durations["optimize_signals"] = 0.0
        durations["analytics"] = 0.0

    durations["total"] = time.time() - t0
    write_pipeline_metrics(
        durations,
        metadata={
            "horizon_min": args.horizon_min,
            "freq_min": args.freq_min,
            "best_k": best_k,
            "output": args.out,
        },
    )

    spark.stop()
    print("🏁 Hoàn tất pipeline.")

if __name__ == "__main__":
    main()
