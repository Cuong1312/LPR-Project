# Giao Diện Nhận Diện Biển Số Xe (ALPR)

## Giới thiệu
Đây là mã nguồn phần mềm nhận diện biển số xe tự động, phục vụ cho Đồ án tốt nghiệp tại Trường Đại học Xây dựng Hà Nội. Hệ thống sử dụng mô hình YOLOv8 để phát hiện và trích xuất ký tự trên biển số phương tiện giao thông tại Việt Nam.

## Tính năng chính
* Nhận diện hoàn toàn cục bộ (Offline) bằng mô hình YOLO, không phụ thuộc vào API bên thứ ba.
* Hỗ trợ đầu vào đa dạng: Hình ảnh, Video (MP4, AVI).
* Tích hợp thuật toán IOU Tracker để bám vết phương tiện và tự động chọn khung hình có chất lượng tốt nhất để nhận diện chữ.
* Giao diện đồ họa (GUI) xây dựng bằng CustomTkinter, cho phép xem lại lịch sử quét và lưu trữ ảnh cắt của biển số.

## Yêu cầu hệ thống
Phần mềm đã được kiểm thử trên cấu hình:
* Vi xử lý: Intel Core 5 210H (12 luồng)
* Bộ nhớ RAM: 16GB DDR4 Bus 3200
* Hệ điều hành: Windows 11
* Môi trường: Python 3.11 trở lên

## Hướng dẫn cài đặt
1. Tải mã nguồn về máy:
   ```bash
   git clone [https://github.com/Cuong1312/LPR-Project.git](https://github.com/Cuong1312/LPR-Project.git)
   cd LPR-Project