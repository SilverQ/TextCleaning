from math import log
# import math


# 위키독스의 TF-IDF
# https://wikidocs.net/31698
def tf(t, d):
    cnt = sum(1 for v in d if v is t or v == t)
    return cnt


def idf(t, d):
    doc_cnt = len(d)        # 여기서 문제는 generator 사용시 전체 d 입력 불가능, doc_cnt를 받아야 함
    df = 0
    for sentence in d:
        df += t in sentence
    return log(doc_cnt / (df + 1))


def tfidf(t, d):
    return tf(t, d) * idf(t)



"""
def tf(t, d):
    return d.count(t)

def count(self, value):
    'S.count(value) -> integer -- return number of occurrences of value'
    return sum(1 for v in self if v is value or v == value)

def idf(t):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N / (df + 1))

def tfidf(t, d):
    return tf(t,d)* idf(t)
"""
