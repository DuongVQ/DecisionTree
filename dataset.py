from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# 75% để training, còn lại để ktra

from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris_dataset = load_iris()
# print(iris_dataset.target) 
# Phân lớp dữ liệu
#thêm len() để biết số điểm dữ liệu

x_train, x_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, random_state=0)
# print(y_test)

model = DecisionTreeClassifier()
myModel = model.fit(x_train, y_train)
x_new = np.array([[6.0, 3.1, 5.5, 1.8]])
# print(myModel.predict(x_new))
print(myModel.score(x_test, y_test)) # tính độ chính xác