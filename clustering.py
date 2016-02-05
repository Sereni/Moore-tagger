__author__ = 'elmira'

from collections import defaultdict
import datetime
import numpy as np
import re
regDigits = re.compile('\\d', flags=re.U | re.DOTALL)


class Word:
    def __init__(self):
        self.predecessors = defaultdict(int)
        self.successors = defaultdict(int)


def load_bigram_frequencies(path):
    """
    :param path: path to the csv file with tab separated values - word1, word2, frequency
    :return: dict
    """
    d = defaultdict(Word)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lower().strip().split('\t')
            line[1] = regDigits.sub('0', line[1])
            line[3] = regDigits.sub('0', line[3])
            d[line[1]].successors[line[3]] = int(line[0])
            d[line[3]].predecessors[line[1]] = int(line[0])
    return d


def load_word_frequencies(path, bigr):
    '''
    :param path: path to the csv file with tab separated values - word and frequency
    :return: dict {word: frequency}
    '''
    d = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lower().strip().split('\t')
            line[1] = regDigits.sub('0', line[1])
            if line[1] in bigr:
                d[line[1]] = int(line[0])
    return d


def count_word_rank(d):
    """
    sort words from the most frequent to the least frequent
    :param d: dictionary of word frequenies
    :return: array of words sorted by frequency
    """
    return sorted(d, key=lambda k: d[k], reverse=True)


def elements_in_top(suc, top):
    return sum(i in suc for i in top)


def bigr_freqs(w, top):
    return [w[i] for i in top]

# 1grams and 2grams from ruscorpora
PATH_TO_WORD_FILE = '1grams-3.txt'
PATH_TO_BIGRAM_FILE = '2grams-3.txt'

# load data
print('loading 2gram freqs...', datetime.datetime.now())
bigrams = load_bigram_frequencies(PATH_TO_BIGRAM_FILE)
print('loading word freqs...', datetime.datetime.now())
words = load_word_frequencies(PATH_TO_WORD_FILE, bigrams)
w_sum = sum(words[w] for w in words)
ordered = count_word_rank(words)
print('generate data table', datetime.datetime.now())

#features:
# 1) words frequency,
# 2) how many of top 100 frequent words occur after word,
# 3) how many of top 100 frequent words occur before word,
# 4) length of word,
# 5) frequency of top 10 bigrams where words is in the second place,
# 6) frequency of top 10 bigrams where words is in the first place
data = np.array([[words[w]/w_sum, elements_in_top(bigrams[w].successors, ordered[:100]),
                  elements_in_top(bigrams[w].predecessors, ordered[:100]), len(w)] +
                 bigr_freqs(bigrams[w].predecessors, ordered[:10]) +
                 bigr_freqs(bigrams[w].successors, ordered[:10]) for w in ordered])
num_clusters = 256
from sklearn.cluster import KMeans
km = KMeans(n_clusters=num_clusters, init='random', n_init=1,verbose=2)
km.fit_predict(data)

d = defaultdict(list)
for i in zip(km.labels_, ordered):
    d[i[0]].append(i[1])

with open('clusters2.txt', 'w', encoding='utf-8') as f:
    for key in d:
        if len(d[key]) > 1:
            print(key, d[key], file=f)
