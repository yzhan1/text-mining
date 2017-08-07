import os
import re
import pprint
import xlrd
import jieba
import random
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def open_file(path):
    dict, corpus = {}, []
    stopwords = [word.strip() for word in open('data/stopwords.txt', 'r')]
    file = xlrd.open_workbook(path)
    for i in range(1, file.nsheets):
        sheet = file.sheet_by_index(i - 1)
        dict[i] = {}
        dict[i]['group_id'] = i
        dict[i]['group_name'] = sheet.name
        for row_idx in range(1, sheet.nrows):
            row = sheet.row_values(row_idx)
            item = {}
            item['id'] = row_idx
            item['title'] = row[2]
            processed_line = process_line(row[3], stopwords)
            item['desc'] = processed_line
            corpus.append(processed_line)
            item['group_id'], item['group_name'] = i, sheet.name
            dict[i][row_idx] = item
    return dict, corpus


def process_line(line, stopwords):
    newline = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）①②③④)]+", "", line)
    seg_list = jieba.cut(newline)
    res = [word for word in seg_list if word not in stopwords]
    joined_res = ' '.join(res)
    return joined_res


def calc_dist(vecA, vecB):
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def vectorize(corpus):
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    matrix = vectorizer.fit_transform(corpus)
    arr = matrix.toarray()
    tfidf = transformer.fit_transform(arr)
    tfidf_arr = tfidf.toarray()
    return tfidf_arr


def k_means(data, k):
    m = np.shape(data)[0]
    data = list(data)
    clusters = np.mat(np.zeros((m, 2)))
    centroids = random.sample(data, k)
    changed = True
    while changed:
        changed = False
        for i in range(m):
            min_dist = np.inf
            min_idx = -1
            for j in range(k):
                dist = calc_dist(centroids[j], data[i])
                if dist < min_dist:
                    min_dist, min_idx = dist, j
            if clusters[i, 0] != min_idx:
                changed = True
            clusters[i] = min_idx, min_dist ** 2
        for center in range(k):
            pts_in_cluster = data[np.nonzero(clusters[:, 0].A == center)[0]]
            centroids[center] = np.mean(pts_in_cluster, axis=0)
    return centroids, clusters


def show(dataSet, k, centroids, cluster_assessment):
    numSamples, dim = dataSet.shape
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(numSamples):
        markIndex = int(cluster_assessment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], markersize=12)
    plt.show()


if __name__ == '__main__':
    tfidf_arr = []
    if os.path.isfile('./data/matrix.txt.npy'):
        tfidf_arr = np.load('data/matrix.txt.npy')
        print(tfidf_arr)
    else:
        path = 'data/data.xlsx'
        dict, corpus = open_file(path)
        tfidf_arr = vectorize(corpus)
        np.save('data/matrix.txt', tfidf_arr)
    centroids, clusters = k_means(tfidf_arr, 7)
    show(tfidf_arr, 7, centroids, clusters)
