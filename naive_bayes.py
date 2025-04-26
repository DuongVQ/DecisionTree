import os # thư viện os cung cấp các chức năng tương tác với hệ điều hành
import string
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import model_selection # thư viện model_selection cung cấp các công cụ để chia tập dữ liệu thành các tập con cho việc huấn luyện và kiểm tra mô hình
from sklearn.naive_bayes import MultinomialNB # MultinomialNB là một trong những biến thể của Naive Bayes, thường được sử dụng cho các bài toán phân loại văn bản
from sklearn.metrics import classification_report # thư viện classification_report cung cấp các công cụ để đánh giá hiệu suất của mô hình phân loại

x = [] # 1 phần tử của x dạng (filename, text)
y = [] # 1 phần tử của y là label của x

# Đọc dữ liệu từ thư mục
for category in os.listdir('./20_newgroups'):
    for document in os.listdir('./20_newgroups/' + category):
        with open('./20_newgroups/' + category + '/' + document, 'r', encoding='utf-8', errors='ignore') as f:
            x.append((document, f.read()))
            y.append(category)

# Chia dữ liệu thành 2 tập train và test
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)

stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 
             'along', 'already', 'also', 'although', 'always', 'an', 'among', 'amongst', 'amoungst', 'amount', 
             'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 
             'as', 'at', 'back', 'be', 'becane', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 
             'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 
             'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de', 
             'describe', 'detail', 'did', 'do', 'does', 'doing', 'don', 'done', 'down', 'due', 'during', 'each', 
             'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 
             'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 
             'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 
             'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'having', 'he', 'hence', 'her', 'here', 
             'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 
             'hundred', 'т', '1', 'ie', 'if', 'in', 'inc', 'Indeed', 'less', 'interest', 'into', 'is', 'it', 'its', 'itself', 
             'just', 'keep', 'last', 'latter', 'latterly', 'least', '1td', 'made', 'many', 'may', 'may', 'me', 'meanwhile', 
             'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 
             'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 
             'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 
             'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 
             'put', 'rather', 're', 's', 'same', 'see', 'seen', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 
             'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 
             'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 't', 'take', 'ten', 'than', 'that',  
             'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 
             'therefore', 'therein', 'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 
             'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 
             'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 
             'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 
             'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whow', 'whose', 'why', 'will', 'with', 'within', 'without', 
             'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves']

# Xây dựng vốn từ vựng cho tài liệu
vocab = {} # khai báo dict rỗng
for i in range(len(x_train)): # duyệt mảng với kích thước của tập training
    word_list = [] # khai báo list rỗng
    for word in x_train[i][1].split(): # duyệt từng từ trong tài liệu
        word_new = word.strip(string.punctuation).lower() # loại bỏ ký tự đặc biệt ở đầu và cuối
        if(len(word_new) > 2) and (word_new not in stopwords): # nếu từ có độ dài lớn hơn 2 và không nằm trong stopwords
            if word_new in vocab: # nếu từ đã có trong vocab 
                vocab[word_new] += 1 #nhảy đến vocab và tăng biến đếm lên 1
            else: # nếu từ chưa có trong vocab
                vocab[word_new] = 1 # khởi tạo từ mới trong vocab với giá trị 1
            
# đồ thị các từ thu được
num_words = [0 for i in range(max(vocab.values()) + 1)] # khởi tạo mảng với kích thước max(vocab.values()) + 1
freq = [i for i in range(max(vocab.values()) + 1)] 
for key in vocab: # duyệt từ trong vocab
    num_words[vocab[key]] += 1 # tăng biến đếm lên 1 với từ đã có trong vocab
plt.plot(freq, num_words) # vẽ đồ thị với trục x là freq và trục y là num_words
plt.axis([1, 10, 0, 20000]) # thiết lập trục x và y
plt.xlabel('Frequency') # thiết lập nhãn cho trục x
plt.ylabel('No of words') # thiết lập nhãn cho trục y
plt.grid() # thiết lập lưới cho đồ thị
# plt.show() # hiển thị đồ thị



cutoff_freq = 80 # thiết lập ngưỡng tần suất

num_words_above_cutoff = len(vocab) - sum(num_words[:cutoff_freq]) # số từ có tần suất lớn hơn ngưỡng tần suất
print("Số từ có tần suất cao hơn ngưỡng tần suất({}) :".format(cutoff_freq), num_words_above_cutoff) # in ra số từ có tần suất lớn hơn ngưỡng tần suất

# các từ có tần suất >80 đc chọn làm đặc trưng
features = []
for key in vocab: # duyệt từ trong vocab
    if vocab[key] > cutoff_freq: # nếu tần suất của từ lớn hơn ngưỡng tần suất
        features.append(key)
    

# biểu diễn datatrain dạng word vector counts
x_train_dataset = np.zeros((len(x_train), len(features))) 
for i in range(len(x_train)): # duyệt mảng với kích thước của tập training
    word_list = [word.strip(string.punctuation).lower() for word in x_train[i][1].split()] # loại bỏ ký tự đặc biệt ở đầu và cuối
    for word in word_list:  
        if word in features:
            x_train_dataset[i][features.index(word)] += 1
    
# biểu diễn datatest dạng word vector counts
x_test_dataset = np.zeros((len(x_test), len(features)))
for i in range(len(x_test)): # duyệt mảng với kích thước của tập test
    word_list = [word.strip(string.punctuation).lower() for word in x_test[i][1].split()] # loại bỏ ký tự đặc biệt ở đầu và cuối
    for word in word_list:  
        if word in features:
            x_test_dataset[i][features.index(word)] += 1



clf = MultinomialNB() # khởi tạo mô hình MultinomialNB
clf.fit(x_train_dataset, y_train) # huấn luyện mô hình với tập training
y_test_pred = clf.predict(x_test_dataset) # dự đoán nhãn cho tập test
sklearn_score_train = clf.score(x_train_dataset, y_train) # tính toán độ chính xác của mô hình trên tập training
print("Độ chính xác của mô hình trên tập training:", sklearn_score_train) 

sklearn_score_test = clf.score(x_test_dataset, y_test) # tính toán độ chính xác của mô hình trên tập test
print("Độ chính xác của mô hình trên tập test:", sklearn_score_test)
print(classification_report(y_test, y_test_pred)) # in ra báo cáo phân loại