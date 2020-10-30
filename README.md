1. Mô hình ban đầu sử dụng multilayer perceptron
  Số lượng tham số: 8,832
  Độ chính xác trên tập test: 0.8859
  
2. Thêm một lớp convolution vào mạng
  Số lượng tham số: 811,930
  Độ chính xác trên tập test: 0.8988

3. Thêm một lớp max pooling vào mạng
  Số lượng tham số: 209,818
  Độ chính xác trên tập test: 0.9092
  
4. Thêm inception block 
  Số lượng tham số: 211,582
  Độ chính xác trên tập test: 0.9139
  
5. Dùng các residual block
  Số lượng tham số: 211,582
  Độ chính xác trên tập test: 0.9139

Kết luận:
•	Mô hình càng nhiều lớp càng dễ overfitting
•	Các phương pháp chống overfitting như regularization, dropout khi áp dụng cho các mô hình với số lượng tham số nhỏ có ảnh hưởng không đáng kể, thậm chí làm mô hình tệ hơn.
•	Kiến trúc dựa theo residual block của resnet cho kết quả tốt nhất
