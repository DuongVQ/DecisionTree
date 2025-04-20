import numpy as np
import pandas as pd # đọc excel

df = pd.read_csv('', header=None) # thay header = 0,1 thì sẽ thay dòng trên cùng là dòng 0,1 của excel
# print(df[3]) - đọc cột 3
print(df)
df.to_csv('duong.csv') #lưu dạng csv