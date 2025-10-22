# Báo Cáo Đồ Án “Traffic Big Data” – Mẫu Soạn Thảo Word

> **Mục đích:** File này đóng vai trò “sườn” để copy sang Word, dễ bổ sung hình ảnh/bảng biểu và tạo mục lục tự động.  
> **Gợi ý:** Trong Word hãy đặt các tiêu đề bằng Heading 1/2/3, chèn hình trong `results/plots_kmeans/` hoặc bảng `traffic_signal_plan/*.csv` khi cần minh họa.

---

## 1. Giới thiệu (Chương 1)
- **Bối cảnh & Vấn đề:** Trình bày thực trạng ùn tắc giao thông đô thị, khối lượng dữ liệu cảm biến (METR-LA) lớn, nhu cầu dự báo sớm để điều tiết đèn.
- **Mục tiêu dự án:**
  1. Xây dựng pipeline xử lý Big Data bằng Apache Spark (ingest → tiền xử lý → phân cụm → dự báo).
  2. Dự báo xác suất tắc nghẽn 15 phút tới và khuyến nghị điều chỉnh pha xanh đèn giao thông.
  3. Phát triển dashboard/web API phục vụ giám sát realtime và ra quyết định.
- **Phạm vi:** Dữ liệu cảm biến tốc độ (METR-LA) giai đoạn 2012; chạy trên HDFS + Spark local[*]; có thể mở rộng.
- **Đóng góp chính:** Pipeline end-to-end, dashboard trực quan, API realtime, module tối ưu đèn, tích hợp Prometheus/Grafana.

---

## 2. Hạ tầng hệ thống & nguồn dữ liệu (Chương 2)
### 2.1 Kiến trúc triển khai
| Thành phần | Vai trò | Ghi chú triển khai |
|------------|---------|--------------------|
| **HDFS / lưu trữ phân tán** | Lưu dữ liệu thô, kết quả trung gian & cuối | Ví dụ: `hdfs://localhost:9000/results/metr_la/…` |
| **Apache Spark** | Xử lý & ML (Spark SQL, MLlib) | Chạy `python3 main.py`, có thể scale lên cluster |
| **Flask Dashboard** | Trực quan hóa + giao diện realtime | Directory `webapp/`, chạy `python3 webapp/app.py` |
| **Prometheus/Grafana** | Giám sát thời gian pipeline | Thu metrics từ `metrics/pipeline_metrics.prom` |
| **Cron / Airflow** | Tự động hóa lịch chạy | Script `scheduler/traffic_pipeline.sh`, DAG `traffic_pipeline_dag.py` |

### 2.2 Dữ liệu cảm biến (METR-LA)
- **Định dạng:** CSV, mỗi cột là sensor speed (km/h); thời gian `M/d/yyyy H:mm`.
- **Đặc thù:** Thiếu dữ liệu (`NaN`), ngoại lai (≤0, >120); có sai lệch timezone.
- **Quy mô:** Hàng chục triệu bản ghi (~5GB cho đoạn mẫu).
- **Các bước chuẩn hóa ban đầu:** ép timestamp, loại ngoại lai cơ bản, nén Parquet trước khi lưu vào HDFS.
> *Minh họa đề xuất:* Ảnh biểu đồ histogram từ `results/analysis/histogram_avg_speed`.

---

## 3. Pipeline Spark & Machine Learning (Chương 3)
### 3.1 Tiền xử lý (modules/reader.py)
- Chuẩn hóa thời gian, loại bỏ tốc độ không hợp lệ.
- Điền thiếu bằng trung bình theo sensor; tính feature:
  - `avg_speed`, `hour`, `weekday`, `std_speed_hour`.
- Áp dụng lọc ngoại lai Hard++ (giới hạn [5,110] km/h).
- Ghi dữ liệu sạch về `{out}/cleaned_data`.

### 3.2 Phân cụm hành vi giao thông (modules/clustering.py)
- Dùng `VectorAssembler` → KMeans.
- Tự động chọn k (Silhouette trong `[kmin, kmax]`).
- Gán nhãn cụm theo tốc độ trung bình (thấp nhất = “Tắc nghẽn”).
- Xuất biểu đồ scatter/box/kde + file `cluster_distribution`.

### 3.3 Dự báo tắc nghẽn 15 phút (modules/prediction.py)
- Tạo nhãn tương lai bằng lead step (theo horizon/freq).
- Chia train/test, cân bằng lớp (class weighting).
- Pipeline: `VectorAssembler` → `StandardScaler` → `LogisticRegression`.
- CrossValidator (regParam 0/0.01/0.05, elasticNet 0/0.3, 2-fold).
- Lưu Parquet dự báo (`results/predict_15m/`), model pipeline, feature importance.

### 3.4 Phân tích thống kê (modules/analytics.py)
- Sinh các bảng: hourly summary, weekday summary, top_congestion, sensor_activity, heatmap, histogram.
- Output dạng JSON để dashboard đọc.

### 3.5 Tối ưu đèn giao thông (modules/optimization.py)
- Từ xác suất dự báo → tính xác suất TB theo `date/hour`.
- Quy tắc pha xanh: ≥0.7 → +30%, ≥0.5 → +20%, ≥0.3 → +10%, else -10%.
- Xuất CSV (`results/traffic_signal_plan/`) và JSON phục vụ API/dashboard.
> *Hình gợi ý:* screenshot bảng “Tối ưu chu kỳ đèn” trong dashboard.

---

## 4. Ứng dụng Flask & API realtime (Chương 4)
### 4.1 Dashboard (webapp/app.py, templates/index.html)
- **Các thẻ chính:** Tổng quan, Phân cụm, Top giờ tắc, Heatmap, Hoạt động cảm biến, Độ quan trọng đặc trưng, Dự báo 15’, Tối ưu đèn.
- **Thẻ Realtime:** form chọn tuyến + nhập tốc độ → gọi `/api/realtime` hiển thị xác suất, F1, chiến lược đèn.
- **Thẻ Tối ưu chu kỳ đèn:** đọc dữ liệu từ `traffic_signal_plan`, hiển thị top rủi ro và bảng chi tiết (ngày, giờ, xác suất, mẫu, điều chỉnh).
- Nên chèn 2–3 ảnh chụp màn hình minh họa.

### 4.2 API `/api/realtime`
- **Payload tối thiểu:** `avg_speed`. Trường `std_speed_hour` có thể để `"auto"` (hệ thống tự tra).
- **Trường tự động:** `hour`, `weekday`, `timestamp` nếu bỏ trống.
- **Kết quả trả về:** xác suất tắc nghẽn, nhãn dự báo (0/1), chiến lược đèn (mức độ + % kéo dài), phân bố xác suất `[p0, p1]`, thông tin model.
- **Use case:** tích hợp với cảm biến realtime, kết hợp hệ thống điều khiển đèn.

### 4.3 Giám sát & vận hành
- Pipeline ghi metrics: `traffic_pipeline_total_duration_seconds`, `traffic_step_duration_seconds{step="..."}`.
- Sử dụng node_exporter + Prometheus để đọc file `.prom`.
- Cung cấp hướng dẫn trong `monitoring/README.md` + Dashboard Grafana mẫu.

---

## 5. Đánh giá theo thang điểm (Chương 5)
| Tiêu chí | Gợi ý nội dung chứng minh | Ghi chú minh chứng |
|---------|---------------------------|--------------------|
| **Lưu trữ phân tán (2đ)** | Mô tả lưu trữ HDFS, cấu trúc thư mục, lệnh `hdfs dfs -get` | Chèn bảng đường dẫn quan trọng |
| **Xử lý & phân tích (4đ)** | Pipeline Spark (KMeans, Logistic Regression, tối ưu đèn), kết quả AUC/AUPRC | Đưa số liệu cụ thể, bảng so sánh model nếu có |
| **Trực quan hóa (2đ)** | Dashboard Flask, biểu đồ scatter/kde/heatmap, bảng tối ưu đèn | Xuất hình từ `results/` |
| **Sáng tạo & hiệu quả (2đ)** | API realtime, khuyến nghị đèn, Prometheus/Grafana, Cron/Airflow | Mô tả quy trình realtime + tự động hóa |

- **Kết luận chương:** tổng hợp ngắn (3–4 câu) nêu điểm mạnh, hạn chế, hướng mở rộng: thêm dữ liệu thời tiết, Spark Streaming, mô hình nâng cao, tích hợp GIS.

---

## 6. Hướng dẫn vận hành nhanh (Phụ lục hoặc cuối báo cáo)
1. **Chuẩn bị môi trường**
   ```bash
   pip install -r requirements.txt
   export SPARK_MASTER=local[*]
   ```
2. **Chạy pipeline Spark**
   ```bash
   python3 main.py \
     --input hdfs://localhost:9000/metr_la/raw.csv \
     --out hdfs://localhost:9000/results/metr_la \
     --timecol timestamp --horizon_min 15 --freq_min 5
   ```
3. **Đồng bộ kết quả về local**
   ```bash
   mkdir -p results
   hdfs dfs -get -f hdfs://localhost:9000/results/metr_la/* results/
   export TRAFFIC_OUTPUT_DIR="$(pwd)/results"
   ```
4. **Khởi chạy dashboard**
   ```bash
   python3 webapp/app.py
   # mở http://localhost:5000
   ```
5. **Lên lịch chạy tự động**
   - Cron: chỉnh `scheduler/traffic_pipeline.sh`, thêm vào `crontab -e`.
   - Airflow: copy `scheduler/traffic_pipeline_dag.py` vào `dags/`, cập nhật biến môi trường.

---

## Phụ lục gợi ý thêm vào Word
### A. Cấu trúc thư mục chính
```
traffic-bigdata/
├── main.py
├── modules/
│   ├── reader.py
│   ├── clustering.py
│   ├── prediction.py
│   ├── optimization.py
│   ├── analytics.py
│   └── visualize.py
├── webapp/
│   ├── app.py
│   └── templates/
├── scheduler/
├── monitoring/
└── results/   (predict_15m, analysis, plots_kmeans, traffic_signal_plan, ...)
```

### B. Danh sách file kết quả quan trọng
- `results/predict_15m/*.parquet`: Bảng dự báo (timestamp, xác suất, nhãn).
- `results/analysis/*/*.json`: Thống kê hourly/weekday/top_congestion/sensor_activity.
- `results/traffic_signal_plan/*.csv`: Khuyến nghị điều chỉnh đèn (date, hour, avg_prob, strategy, count_samples).
- `metrics/pipeline_metrics.json`: metadata (horizon, freq_min, best_k) và thời gian từng bước.

### C. API realtime – ví dụ request/response
```json
POST /api/realtime
{
  "avg_speed": 42.5,
  "std_speed_hour": "auto",
  "hour": 17,
  "weekday": 2,
  "latitude": 10.8019,
  "longitude": 106.7123,
  "location": "Nguyễn Văn Trỗi - Nam Kỳ"
}
```
Response trả về:
```json
{
  "prob_congestion": 0.62,
  "prediction": 1,
  "strategy": {
    "level": "high",
    "green_extension_pct": 20,
    "strategy": "Kéo dài pha xanh thêm 20%"
  },
  "probability_distribution": [0.38, 0.62],
  "inputs": { ... },
  "model": { "path": ".../logistic_pipeline", "last_modified": "..." }
}
```

### D. Hướng phát triển (đưa vào phần kết luận)
- Bổ sung dữ liệu thời tiết, sự kiện để cải thiện dự báo.
- Áp dụng Spark Structured Streaming cho realtime pipeline.
- Thử nghiệm mô hình nâng cao (Gradient Boosting, LSTM, Graph Neural Network).
- Chuẩn hóa API để tích hợp với hệ thống điều khiển đèn thực tế, hỗ trợ nhiều tuyến đường.

---

> **Kết thúc:** Khi copy bản “sườn” này sang Word, chỉ cần thêm hình ảnh (từ thư mục `results/`), bảng số liệu (CSV/JSON), viết đoạn kết luận & lời cảm ơn. Tài liệu sẽ đầy đủ và dễ chấm theo thang điểm của đề bài. Chúc bạn hoàn thành đồ án thật tốt! 💪
