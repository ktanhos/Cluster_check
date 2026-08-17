# VN30 Rolling Market Behavior Clustering

Nghiên cứu phân cụm hành vi VN30 theo chuỗi thời gian rolling, sử dụng dữ liệu giá và khối lượng lấy trực tiếp từ VNstock.

## Ý tưởng

Thay vì phân cụm một ảnh chụp duy nhất của 30 cổ phiếu, hệ thống dùng cửa sổ quá khứ để xây dựng trạng thái và phân loại 30 cổ phiếu tại từng mốc thời gian.

Quy trình:

VNstock → cache dữ liệu → feature rolling → rolling KMeans → cluster state → migration → forward return event study.

## Sáu đặc trưng

Return20, Volatility20, Beta60, RS20, VolumeZ20 và DistanceVN60.

## Mặc định

Khoảng nghiên cứu: 01/03/2026 đến 14/08/2026.

Cửa sổ rolling: 60 phiên.

K: 4.

Bước cập nhật: 5 phiên.

## Chạy ứng dụng

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dữ liệu tải về được lưu trong `data_cache` để hạn chế gọi lại nguồn dữ liệu. Ứng dụng không gọi API cho từng ngày; mỗi mã được tải một lần cho toàn bộ khoảng lịch sử cần thiết.

## Lưu ý phương pháp

K = 4 chỉ là cấu hình mặc định kế thừa từ thử nghiệm trước. Không coi 4 nhóm là một cấu trúc cố định của thị trường cho đến khi kiểm tra độ ổn định theo thời gian.

Forward Return trong phiên bản này là event study ban đầu, chưa phải kiểm định alpha hoàn chỉnh. Cần bổ sung kiểm định ngoài mẫu, ý nghĩa thống kê, data snooping và chi phí giao dịch trước khi kết luận về khả năng dự báo.
