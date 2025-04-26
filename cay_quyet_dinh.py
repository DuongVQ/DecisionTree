'''
Quy ước đầu vào:
nhẹ: 1
thấp: 2
trung bình: 3
cao: 4
nặng: 5
ít: 6
nhiều: 7

******************

Quy ước đầu ra:
có: 1
không: 0

'''

from sklearn import tree

# Bước 1: thu thập dữ liệu
# Bước 2: xử lý dữ liệu
# Bước 3: xây dựng model
# Bước 4: dự đoán kết quả
# Bước 5: đánh giá
my_tree = tree.DecisionTreeClassifier()
'''
DecisionTreeClassifier:
- dùng cho bài toán phân loại, 
- dự đoán nhóm, nhãn
- đầu ra là các giá trị rời rạc
- hàm mất mát: entropy, gini impurity
- kết quả dự đoán: predict() trả về nhãn

DecisionTreeRegressor:
- dùng cho bài toán quy hồi
- dự đoán giá trị số liên tục 
- đầu ra là các giá trị thực
- hàm mất mát: mse, mae
- kết quả dự đoán: predict() trả về giá trị số
'''

feature = [
    [1, 3, 3, 7],
    [5, 2, 4, 6],
    [1, 2, 4, 6],
    [5, 4, 4, 3],
    [1, 4, 4, 7],
    [3, 2, 3, 7],
    [3, 3, 3, 6],
    [5, 2, 2, 7]
]

label = [0, 1, 1, 0, 0, 0, 0, 1]

result = my_tree.fit(feature, label) # fit() dùng để huấn luyện mô hình

kq = result.predict([
    [1, 4, 3, 6],
    [1, 4, 3, 7]
]) # predict() dùng để dự đoán kết quả
print(kq)
print("Dự đoán bệnh tim: ")
for x in kq:
    if x == 1:
        print("Có")
    else:
        print("Không")


'''
Entropy: đo mức độ đồng nhất của một tập hợp các nhãn
- nếu tất cả các nhãn đồng nhất thì entropy = 0
- nếu các mẫu chia đều thì entropy = 1

        Entropy = -p1*log2(p1) - p2*log2(p2) - ... - pn*log2(pn)

- https://www.saedsayad.com/decision_tree.htm
'''