## Giám sát pipeline với Prometheus/Grafana

### 1. Thu thập metric

Pipeline viết thông tin thời gian chạy vào file `metrics/pipeline_metrics.prom` theo định dạng Prometheus textfile. Để Prometheus đọc được:

1. Cài `node_exporter` và bật textfile collector, ví dụ:
   ```bash
   ./node_exporter --collector.textfile.directory=/home/lntuandat/traffic-bigdata/metrics
   ```
2. Kiểm tra file metric sau khi pipeline chạy:
   ```bash
   cat metrics/pipeline_metrics.prom
   ```
   Mẫu nội dung:
   ```
   traffic_pipeline_last_run_timestamp_seconds 1700000000
   traffic_pipeline_total_duration_seconds 182.40
   traffic_step_duration_seconds{step="read_and_clean"} 46.32
   ...
   ```

### 2. Cấu hình Prometheus

Thêm vào `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: "traffic-pipeline"
    static_configs:
      - targets: ["localhost:9100"]  # node_exporter
```

### 3. Dashboard Grafana

Import dữ liệu từ Prometheus sau đó tạo dashboard với các biểu đồ:

- **Thời gian tổng**: biểu đồ line cho `traffic_pipeline_total_duration_seconds`.
- **Chi tiết từng bước**: `traffic_step_duration_seconds` grouped by label `step`.
- **Tần suất chạy**: panel dạng singlestat hiển thị `time() - traffic_pipeline_last_run_timestamp_seconds`.

Có thể dùng JSON từ `metrics/pipeline_metrics.json` để kiểm tra nhanh hoặc hiển thị trên Flask.

### 4. Độ trễ end-to-end

Sử dụng metric `traffic_pipeline_total_duration_seconds` và, nếu cần, thêm PromQL:
```promql
rate(traffic_pipeline_total_duration_seconds[6h])
```
để quan sát xu hướng.

### 5. Báo động

Cấu hình alert nếu pipeline quá 30 phút:
```yaml
- alert: TrafficPipelineSlow
  expr: traffic_pipeline_total_duration_seconds > 1800
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Pipeline xử lý giao thông đang chậm"
```

Như vậy toàn bộ vòng đời pipeline được giám sát trực quan trên Grafana.
