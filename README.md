# VN30 Rolling Market Behavior Clustering

Nghiên cứu phân cụm hành vi VN30 theo chuỗi thời gian rolling, sử dụng dữ liệu giá và khối lượng lấy trực tiếp từ VNstock.

## Kiến trúc nghiên cứu

Thay vì phân cụm một ảnh chụp duy nhất của 30 cổ phiếu, hệ thống dùng cửa sổ quá khứ để xây dựng trạng thái và phân loại 30 cổ phiếu tại từng mốc thời gian.

Quy trình:

VNstock → cache dữ liệu → feature rolling → chuẩn hóa chéo theo ngày → rolling KMeans → ổn định State ID → migration → forward return event study.

## Sáu đặc trưng

Return20, Volatility20, Beta60, RS20, VolumeZ20 và DistanceVN60.

Các đặc trưng được chuẩn hóa theo từng ngày trên toàn bộ VN30 trước khi đưa vào KMeans. Cách này giữ cho hình học của feature space không thay đổi chỉ vì StandardScaler được tái ước lượng ở mỗi rolling window.

## Ổn định Cluster ID

KMeans không đảm bảo Cluster 0 ở hai cửa sổ liên tiếp biểu diễn cùng một trạng thái. Hệ thống dùng thuật toán ghép tối ưu giữa centroid của hai cửa sổ liên tiếp để ánh xạ nhãn mới về State ID ổn định.

Migration được lưu ở hai mức:

Migration thô: bất kỳ thay đổi State ID nào.

Migration xác nhận: trạng thái mới phải duy trì đủ số mốc cấu hình và có Assignment Confidence đạt ngưỡng.

## Assignment Confidence

Khoảng cách tới centroid gần nhất được so sánh với khoảng cách tới centroid gần thứ hai. Confidence cao nghĩa là cổ phiếu nằm rõ trong một trạng thái; confidence thấp nghĩa là cổ phiếu nằm gần ranh giới giữa hai trạng thái.

## Mặc định

Khoảng nghiên cứu: 01/03/2026 đến 14/08/2026.

Cửa sổ rolling: 60 phiên.

K: 4.

Bước cập nhật: 5 phiên.

Xác nhận migration: 2 mốc.

Ngưỡng confidence: 0,10.

## Tối ưu dữ liệu VNstock

Ứng dụng không gọi API cho từng ngày. Mỗi mã được tải theo khoảng lịch sử cần thiết và lưu vào `data_cache`.

Nếu cache đã có dữ liệu, hệ thống chỉ gọi phần dữ liệu còn thiếu ở đầu hoặc cuối khoảng thời gian rồi hợp nhất vào cache. Điều này tránh tải lại toàn bộ lịch sử trong mỗi lần chạy.

## Chạy ứng dụng

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Lưu ý phương pháp

K = 4 chỉ là cấu hình mặc định kế thừa từ thử nghiệm trước. Không coi 4 nhóm là một cấu trúc cố định của thị trường cho đến khi kiểm tra độ ổn định theo thời gian.

Forward Return trong phiên bản này là event study ban đầu, chưa phải kiểm định alpha hoàn chỉnh. Cần bổ sung kiểm định ngoài mẫu, ý nghĩa thống kê, data snooping và chi phí giao dịch trước khi kết luận về khả năng dự báo.
