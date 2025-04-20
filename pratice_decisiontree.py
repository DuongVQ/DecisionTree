import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Giả lập dữ liệu đầu vào
data = {
    'gia_dau_tho': [60, 70, 65, 80, 90, 85, 100, 95, 75, 88],
    'ti_gia_usd': [23000, 23100, 23200, 22900, 22800, 23300, 23400, 23000, 23250, 23150],
    'thue': [10, 10, 10, 10, 10, 12, 12, 12, 11, 11],
    'chi_phi_van_chuyen': [2, 2.5, 2.2, 2.8, 3, 2.6, 2.9, 3.1, 2.4, 2.7],
    'gia_xang': [18, 19.5, 19, 21, 22.5, 22, 24, 23.5, 20, 21.8]
}

# Chuyển dữ liệu thành DataFrame
df = pd.DataFrame(data)

# Tách dữ liệu thành X và y
X = df[['gia_dau_tho', 'ti_gia_usd', 'thue', 'chi_phi_van_chuyen']]
y = df['gia_xang']

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo model Decision Tree
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Thử dự đoán giá xăng mới
new_input = pd.DataFrame({
    'gia_dau_tho': [92],
    'ti_gia_usd': [23200],
    'thue': [11],
    'chi_phi_van_chuyen': [2.9]
})
du_doan = model.predict(new_input)
print("Giá xăng dự đoán:", round(du_doan[0], 2), "nghìn đồng/lít")
