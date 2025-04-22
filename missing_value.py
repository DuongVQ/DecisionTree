import numpy as np
import pandas as pd
# dùng để thay thế giá trị bị thiếu trong dữ liệu   
from sklearn.impute import SimpleImputer
# lên phiên bản mới rồi nên ko import imputer đc

# đọc file excel
data = pd.read_csv('Book1.csv', header=None, sep=';')

# khai báo giá trị bị thiếu (trong five excel là NaN) và thay thế bằng giá trị trung bình của cột
# mean: giá trị trung bình
# median: giá trị trung vị
# most_frequent: giá trị xuất hiện nhiều nhất
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data)
result = imp.transform(data) 
print(result)