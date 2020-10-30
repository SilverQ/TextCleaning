# coding=utf8
import os
from collections import Counter, defaultdict
from math import log  # IDF 계산을 위해
import pickle
from utils import *
from wordcloud import WordCloud
import nltk
from nltk import regexp_tokenize    # 영어 토크나이저
from konlpy.tag import Okt
from nltk.corpus import stopwords   # 한글 토크나이저
from matplotlib.animation import FuncAnimation
# import numpy as np
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# import spacy


# konlpy 설치
# https://ellun.tistory.com/46

t = Okt()

title = []
for i, doc in enumerate(read_xlsx('sna_v2.xlsx', 'rawdata')):
    title.append(doc)

print(doc[:10])
