import re
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LSHForest

file_path = 'data/videotitle'
stopwords_path = 'data/stopwords.txt'

stopwords = [word.strip() for word in open(stopwords_path, 'r')]

tfidf_vectorizer = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 20), use_idf=1, smooth_idf=1,sublinear_tf=1)
train_documents = []
title_hash = {}

with open(file_path, 'rb') as f:
    for i, line in enumerate(f):
        line = line.decode('gbk')
        title_hash[i] = line
        newline = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）①②③④)]+", "", line)

        # Segment each line and remove stopwords
        seg_list = jieba.cut(newline)
        remove_stopwords = [word for word in seg_list if word not in stopwords]
        # print(seg_list)
        res = ''.join(remove_stopwords)
        # res = [word.lower() for word in seg_list]
        # print(res)
        train_documents.append(res)

x_train = tfidf_vectorizer.fit_transform(train_documents)

test_data_1 = '我想问一下我想离婚他不想离，孩子他说不要'
test_cut_raw_1 = jieba.cut(test_data_1)
test_remove_stopwords = [word for word in test_cut_raw_1 if word not in stopwords]
test = ''.join(test_remove_stopwords)
# print(test)
x_test = tfidf_vectorizer.transform([test])

lshf = LSHForest(random_state=42)
lshf.fit(x_train.toarray())

distances, indices = lshf.kneighbors(x_test.toarray(), n_neighbors=10)
for lst in indices:
    for num in lst:
        print(title_hash[num])