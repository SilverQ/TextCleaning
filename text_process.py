# coding=utf8
import os
from collections import Counter, defaultdict
from math import log  # IDF 계산을 위해
import pickle
from utils import *
from wordcloud import WordCloud
from konlpy.tag import Okt
import nltk
from nltk import regexp_tokenize    # 영어 토크나이저
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer     # 한글 토크나이저
# import numpy as np
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import spacy

# konlpy 설치
# https://ellun.tistory.com/46
# Penn Treebank Tokenization 규칙 - 하이픈으로 구성된 단어는 하나로 유지, doesn't와 같이 어퍼스트로피로 접어가 함께하는 단어는 분
t = Okt()

df = pd.read_excel('sna_v2.xlsx', sheet_name='rawdata')
# df = pd.read_excel('sna_v2_10.xlsx', sheet_name='rawdata', encoding='utf-8')
df = df.set_index('Publication Number-발행번호')
print('Success to load Excel file with', len(df), 'patents')
set_pandas_display_options()

# print(df.columns)
# print(df.head(1))

df_ko = df[df['발행국'] == 'KIPO']
tokenizer = TreebankWordTokenizer()
print('Success to identify KR', len(df_ko), 'patents')

df_us = df[df['발행국'] == 'USPTO']
print('Success to identify US', len(df_us), 'patents')

title_np = np.array(df_us['발명의명칭'])
# print(df_us.describe())     # 3334건
vec = CountVectorizer(min_df=1, encoding='utf-8', stop_words='english')
title_dtm = vec.fit_transform(title_np)     # return document-term matrix
print('Success to create Term-Doc Matrix')
title_tdm = pd.DataFrame(title_dtm.toarray(), index=df_us.index,
                         columns=vec.get_feature_names(), dtype='d')
# print(title_tdm.head(5))    # [3334 rows x 2232 columns] -> stopword 지정 후 [5 rows x 2155 columns]

num_compo = 2
# vec.get_feature_names() : term-doc matrix의 컬럼명이 될 단어들
lsa = TruncatedSVD(num_compo, algorithm='arpack')   # randomized algorithm
dtm_lsa = lsa.fit_transform(title_tdm)
# print(dtm_lsa)
dtm_lsa = pd.DataFrame(Normalizer(copy=False).fit_transform(dtm_lsa), index=df_us.index)
print(dtm_lsa.head())   # 주성분 분해가 완료된 문서 출력

# # 주성분만 출력해보기
# dtm_compo = pd.DataFrame(lsa.components_, index=[i for i in range(num_compo)],
#                          columns=vec.get_feature_names())
# print(dtm_compo)
#
# xs = [dtm_lsa.loc[i, 0] for i in dtm_lsa.index]
# ys = [dtm_lsa.loc[i, 1] for i in dtm_lsa.index]
# # ys = [w[1] for w in dtm_lsa]
#
# plt.figure()
# plt.scatter(xs, ys)
# plt.show()

# Calculate Similarity
similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
sim_pd = pd.DataFrame(similarity, index=df_us.index, columns=df_us.index)
# print(sim_pd.head(5))

sim_idx_list = np.array(sim_pd.get('US007976648B1'))
print(sim_idx_list)
# sim_idx_list.argsort()[::-1]
print(sim_idx_list[sim_idx_list.argsort()[::-1]])
# print(sim_pd['US007976648B1'].index, np.array(sim_pd['US007976648B1']))
