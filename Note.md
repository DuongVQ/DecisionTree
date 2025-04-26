# Machine learning là gì?
- Là một nhánh của AI, gồm các thuật toán giúp máy tính có thể học hỏi dữ liệu để giải quyết vấn đề cụ thể

# Numppy
- Là thư viện tính toán trên mảng
- ndim: in số chiều của mảng
>>> a
array([1, 3, 4, 5, 3, 8])
>>> a[1:3]
array([3, 4])
=> Lấy từ vị trí bắt đầu đến vị trí kết thúc - 1

- matplotlib dùng để vẽ biểu xét xét tập hợp các điểm dữ liệu rồi phân lớp chúng
- vd khi ta có rất nhiều điểm dữ liệu, thông tin của 1 ng là 1 điểm

# Phương sai
- Là 1 gtri đại diện cho độ phân tán của các số liuej so với giá trị trung bình 
- tính: np.var(arr, ddof=1)
- độ lệch chuẩn = căn phương sai: sqrt(phuong_sai)

#  Overfitting và Underfitting Regularization và cross validation machine learning
- Overfitting là hiện tượng mô hình (thuật toán) đạt kết quả tốt trên tập data huấn luyện nhưng kém trên tập data thực tế
 => tăng lượng data training
    giảm biến feature
    tăng độ lớn của parameter chuẩn hóa
- Underfitting là hiện tượng học ngu kém cả 2
 => tìm biến feature khác
    them biến feature dạng (x1^2, x2^2, x1, x2)
    giảm paramter xuống
    