__author__ = 'e'
import json
import csv
"""Просто две функции, чтобы добавить номера кластеров в фичи"""


def read_clusters(fname):
    d = {}
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            class_num = ''
            for char in line:
                if char == ' ':
                    break
                else:
                    class_num += char
            arr = line.lstrip(class_num+' ').rstrip('\r\n').replace("'", '"')
            arr = json.loads(arr)
            for w in arr:
                d[w] = class_num
    return d


def add2features(fname, d):
    with open(fname, 'r', encoding='utf-8') as f, open('feature_matrix_clusters.csv', 'w', encoding='utf-8') as feat:
        reader = csv.reader(f, delimiter=';')
        next(reader, None)  # skip header
        header = 'token;token_lower;cluster;capital;digit;hyphen;prefix1;prefix2;prefix3;prefix4;suffix1;suffix2;suffix3;suffix4;shape1;shape2;POS'
        print(header, file=feat)
        for row in reader:
            if row[1] in d:
                class_num = d[row[1]]
            else:
                class_num = '256'
            row[2:2] = [class_num]
            print(';'.join(row), file=feat)

d = read_clusters('clusters2.txt')
add2features('feature_matrix.csv', d)