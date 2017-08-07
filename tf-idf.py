import math
import pprint

title_hash, count_hash, res = {}, {}, {}
num_of_words, num_of_docs = 0, 0
segmented_lst = []
lst = open('data/allType.txt', 'r').read().splitlines()

print('### Processing')
for i, line in enumerate(lst):
    line = line.split(' ')[1:]
    title_hash[i] = line
    num_of_docs += 1
    segmented_lst.append(line)
    for word in line:
        if word == 'nbsp':
            continue
        if word in count_hash:
            count_hash[word]['total'] += 1
            if i in count_hash[word]:
                count_hash[word][i] += 1
            else:
                count_hash[word][i] = 1
        else:
            count_hash[word] = {}
            count_hash[word]['total'] = 1
            count_hash[word][i] = 1
        num_of_words += 1

print('### Calculating TF-IDF values for documents')
for i, lst in enumerate(segmented_lst):
    vec = []
    for word in lst:
        tf_idf = 0.0
        if word in count_hash:
            if count_hash[word]['total'] >= 5:
                tf = count_hash[word][i] / len(title_hash[i])
                tidj = len(count_hash[word]) - 1
                idf = math.log(num_of_docs / (1 + tidj))
                tf_idf = tf * idf
        vec.append(tf_idf)
    res[i] = vec

pprint.pprint(res)