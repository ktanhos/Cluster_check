# VN30 Hành vi thị trường

Ứng dụng nghiên cứu cách các cổ phiếu VN30 vận động giống hoặc khác nhau theo thời gian. Hệ thống chỉ sử dụng dữ liệu giá và khối lượng, sau đó tự chia cổ phiếu thành các nhóm hành vi.

## Dành cho ai

Giao diện chính được thiết kế cho nhà đầu tư không chuyên về tài chính định lượng. Các thuật ngữ kỹ thuật vẫn được giữ trong phần thông tin nâng cao để người nghiên cứu có thể kiểm tra phương pháp.

## Cách sử dụng

1. Nhập mã truy cập VNstock.
2. Chọn khoảng thời gian nghiên cứu.
3. Bấm `Tải dữ liệu mới`.
4. Chọn cách nhận diện nhóm. Có thể giữ nguyên mặc định nếu chưa nghiên cứu sâu.
5. Bấm `Phân tích hành vi`.
6. Đọc ba biểu đồ chính: Bản đồ hành vi hiện tại, Các nhóm thay đổi như thế nào và Cổ phiếu chuyển nhóm theo thời gian.
7. Cuối cùng xem bảng lợi suất sau khi chuyển nhóm để kiểm tra xem sự thay đổi hành vi có liên quan đến lợi suất tương lai hay không.

## Ý nghĩa các thiết lập

`Số phiên dùng để nhận diện hành vi`: khoảng thời gian gần nhất dùng để mô tả cách cổ phiếu đang vận động. Mặc định 60 phiên, xấp xỉ 3 tháng giao dịch.

`Số nhóm hành vi`: số nhóm mà thuật toán sẽ tạo. Mặc định 4. Đây là lựa chọn nghiên cứu, không có nghĩa thị trường luôn tồn tại đúng 4 nhóm.

`Cứ bao nhiêu phiên cập nhật một lần`: tần suất chụp lại trạng thái. Mặc định 5 phiên.

`Một lần chuyển nhóm cần được giữ trong bao nhiêu lần quan sát`: bộ lọc để tránh coi một thay đổi rất ngắn là chuyển nhóm thực sự.

`Độ chắc chắn tối thiểu khi gán nhóm`: mức độ tách biệt của một cổ phiếu với nhóm gần thứ hai. Có thể giữ mặc định nếu không nghiên cứu sâu.

## Sáu thông tin dùng để nhận diện hành vi

Lợi suất 20 phiên, mức biến động 20 phiên, độ nhạy với VN Index trong 60 phiên, sức mạnh tương đối 20 phiên, mức bất thường của khối lượng 20 phiên và mức khác biệt so với VN Index trong 60 phiên.

Các thông tin này được chuẩn hóa trước khi phân nhóm. Các tâm nhóm giữa các lần quan sát cũng được liên kết để hạn chế việc tên nhóm thay đổi chỉ vì thuật toán đánh số lại nhóm.

## Ba câu hỏi chính của ứng dụng

`Bản đồ hành vi hiện tại`: Những cổ phiếu nào đang vận động giống nhau?

`Các nhóm thay đổi như thế nào`: Quy mô từng nhóm đang mở rộng hay thu hẹp?

`Cổ phiếu chuyển nhóm theo thời gian`: Cổ phiếu nào thay đổi trạng thái và thay đổi đó có bền hơn hay không?

## Kiểm tra khả năng dự báo

Phần lợi suất sau khi chuyển nhóm là một nghiên cứu sự kiện ban đầu. Nó chưa phải kiểm định một chiến lược đầu tư hoàn chỉnh. Cần tiếp tục kiểm định ngoài mẫu, ý nghĩa thống kê, hiện tượng thử quá nhiều giả thuyết và chi phí giao dịch trước khi kết luận về khả năng dự báo.

## Dữ liệu

Khi bấm `Tải dữ liệu mới`, ứng dụng lấy một bộ dữ liệu mới từ VNstock và giữ dữ liệu trong phiên Streamlit. Thay đổi thiết lập mô hình sau đó không gọi VNstock lại.

## Chạy ứng dụng

```bash
pip install -r requirements.txt
streamlit run app.py
```
