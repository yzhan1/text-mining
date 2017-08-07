import re
import jieba
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import LSHForest

file_path = 'data/videotitle'
stopwords_path = 'data/stopwords.txt'

stopwords = [word.strip() for word in open(stopwords_path, 'r')]

tfidf_vectorizer = TfidfVectorizer(min_df=3, max_features=None, ngram_range=(1, 20), use_idf=1, smooth_idf=1,sublinear_tf=1)
joined_docs = []
corpora_docs = []
title_hash = {}

print('########## Building corpora documents')
with open(file_path, 'rb') as f:
    for i, line in enumerate(f):
        line = line.decode('gbk')
        title_hash[i] = line
        newline = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）①②③④)]+", "", line)

        # Segment each line and remove stopwords
        seg_list = jieba.cut(newline)
        res = [word for word in seg_list if word not in stopwords]
        joined_res = ' '.join(res)
        # print(seg_list)
        # res = [word for word in seg_list]

        document = TaggedDocument(words=res, tags=[i])
        corpora_docs.append(document)
        joined_docs.append(joined_res)

print('########## Training Doc2Vec model')
model = Doc2Vec(size=500, min_count=1, iter=20)
model.build_vocab(corpora_docs)
model.train(corpora_docs, total_examples=model.corpus_count, epochs=model.iter)
model.save('doc2vec.model')

print('########## Training tf-idf model')
x_train = tfidf_vectorizer.fit_transform(joined_docs)


print('########## Pre-processing test data')
test_data = '家人因涉嫌运输毒品被抓, 只是去朋友家探望朋友'
test_cut_raw = jieba.cut(test_data)
test = [word for word in test_cut_raw if word not in stopwords]
joined_test = ''.join(test)
inferred_vector = model.infer_vector(test)
x_test = tfidf_vectorizer.transform([joined_test])

print('########## Fetching most similar documents:')
sims = model.docvecs.most_similar([inferred_vector], topn=10)

lshf = LSHForest(random_state=42)
lshf.fit(x_train.toarray())

distances, indices = lshf.kneighbors(x_test.toarray(), n_neighbors=10)

print('Test data: ', test_data)
print('########## Results returned by doc2vec:')
for item in sims:
    print(title_hash[item[0]])

print('########## Results returned by tfidf:')
for lst in indices:
    for num in lst:
        print(title_hash[num])