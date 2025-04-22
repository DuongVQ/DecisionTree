'''
Hồi quy hồi tuyến tính (Linear Regression) 
- Là 1 thuật toán học có giám sát(nhãn), đầu ra dự đoán là liên tục và độ dốc ko đổi

- Đc sử dụng để dự đoán các giá trị trong phạm vị liên tục thay vì cố phân loại chúng
thành các danh mục

- Có 2 loại chính:
  + Simple Linear Regression: hồi quy tuyến tính đơn giản, chỉ có 1 biến độc lập
                y = mx + b
    trong đó:
        y: biến phụ thuộc (đầu ra)
        x: biến độc lập (đầu vào)
        m: hệ số góc (độ dốc) của đường thẳng hồi quy
        b: hệ số tự do (điểm cắt trục y)

    ví dụ:
    company   product($)   sales
    A         10.4         100
    B         12.8         97.1
    C         15.6         95.5

    sales = weight.product + bias
    trong đó:
        product: biến độc lập (feature)
        weight: hệ số của biến độc lập
        bias: giá trị lệch để bù đắp sai số
    tìm kiếm weight và bias để có 1 đg thẳng phù hợp nhất với dữ liệu
  + Multiple Linear Regression: hồi quy tuyến tính bội, có nhiều biến độc lập

- Hàm chi phí: để tối ưu weight
- Hàm lỗi MSE (Mean Squared Error): để đo sự sai khác bằng cách lấy tbinh bình phương giữa giá trị thực tế và giá trị dự đoán
                              n
                MSE = (1/n) * Σ(y_i - (mx_i + b))^2  ||| dùng bình phương để loại bỏ dấu âm
                             i=1 
    trong đó:
        n: số lượng mẫu
        y_i: giá trị thực tế của mẫu i, lấy từ các nhãn đã biết
        mx_i + b: giá trị dự đoán của mẫu i
        Σ: tổng các giá trị từ 1 đến n
'''

import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('Advertising.csv', sep=';')

# cần giá trị cột radio để thí nghiệm và sales làm nhãn
radio = dataframe['Radio'].values
sales = dataframe['Sales'].values

# plt.scatter(radio, sales, marker='o')
# plt.show()

# hàm dự đoán
def predict(new_radio, weight, bias):
    return new_radio * weight + bias

# hàm tính toán chi phí
def cost_function(radio, sales, weight, bias):
    n = len(radio) # tính trung bình của n mẫu
    total_error = 0 # tổng sai số
    for i in range(n):
        # tổng lỗi = giá trị thực tế - giá trị dự đoán
        total_error += (sales[i] - predict(radio[i], weight, bias)) ** 2
    return total_error / n # trả về giá trị trung bình của tổng lỗi

# hàm update weight và bias
def update_weight(radio, sales, weight, bias, learning_rate):
    n = len(radio) # số lượng mẫu
    weight_gradient = 0 # gradient của weight
    bias_gradient = 0 # gradient của bias

    for i in range(n):
        # tính toán gradient cho weight và bias
        weight_gradient += -2*radio[i] * (sales[i] - predict(radio[i], weight, bias))
        bias_gradient += -2*(sales[i] - predict(radio[i], weight, bias))

    # cập nhật weight và bias
    weight -= (weight_gradient/n) * learning_rate
    bias -= (bias_gradient/n) * learning_rate 

    return weight, bias

# hàm train
def train(radio, sales, weight, bias, learning_rate, iter):
    # lưu trữ giá trị chi phí qua các lần lặp
    cost_history = [] 

    # lặp lại quá trình cập nhật weight và bias
    for i in range(iter):
        weight, bias = update_weight(radio, sales, weight, bias, learning_rate)
        cost = cost_function(radio, sales, weight, bias) # tính toán chi phí
        cost_history.append(cost) # lưu trữ chi phí vào danh sách
    
    return weight, bias, cost_history

weight, bias, cost = train(radio, sales, 0.03, 0.0014, 0.001, 60) #số lần lặp là 30
print("Weight:", weight)
print("Bias:", bias)
print("Cost:", cost)
print("Predicted Sales:", predict(10, weight, bias)) 

solanlap = [i for i in range(60)] 
plt.plot(solanlap, cost)
plt.show()

# tăng số lần lặp lỗi sẽ giảm đi