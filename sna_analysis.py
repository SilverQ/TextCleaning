# coding=utf8
import os
from collections import Counter, defaultdict
from math import log  # IDF 계산을 위해
import pickle
from utils import *
from wordcloud import WordCloud
from konlpy.tag import Twitter
from nltk.corpus import stopwords   # 한글 토크나이저
# import nltk
# from nltk import regexp_tokenize    # 영어 토크나이저
# from matplotlib.animation import FuncAnimation
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import spacy

# 파이썬으로 영어와 한국어 텍스트 다루기
# https://www.lucypark.kr/courses/2015-dm/text-mining.html
# NetworkX 모듈을 이용해서 SNA 차트 생성
# https://networkx.org/documentation/stable/reference/algorithms/bipartite.html

stop_set = set(stopwords.words('english'))
# print(len(stop_set))    # 179
# 불용어 목록은 여기서 확인
# C:\Users\Administrator\AppData\Roaming\nltk_data\corpora\stopwords
# stop_kor = set(stopwords.words('korean'))
# print(stop_kor)

"""
    품목통계 데이터에서 발명의 명칭을 사용하여, 특허 사이의 클러스터를 생성하고 유사성을 파악해보자
    발명의 명칭에서 단어를 추출하여 출원번호-단어 테이블을 생성하고, bipartite 한다.

    Step1: 데이터 입력
      입력할 데이터 형태 - 가. 번호와 Document를 파일로 입력
                        - 나. 번호만 입력하면 DB에서 추출
      수집할 데이터 컬럼 - [출원번호], [기준(출원)연도], [텍스트(발명의 명칭, 요약, 청구항 중 1개 이상 취사 선택)]

    Step2: Tokenizing
    Step3: Regularization
    Step4: Pos-tagging
    Step5: 사전 구축
    Step6: 지표 산출(TF-IDF)
    
    Step7: Selecting Words with IDF and POS info.
    
"""

vocab_fn = "vocab.pickle"
word_freq_fn = 'freq_word.pickle'
doc_freq_fn = 'freq_doc.pickle'
data_dir = 'metadata'
img_dir = 'img'

t = Twitter()


def word_freq():
    freq_word_path = os.path.join(data_dir, word_freq_fn)
    try:
        with open(freq_word_path, 'rb') as freq_word_f:
            freq_word = pickle.load(freq_word_f)
            print('word frequency distribution loaded')
        docs_cnt = len(freq_word)
        return docs_cnt, freq_word
    except IOError:
        pass

    print('building frequency distribution')
    freq = defaultdict(int)
    for i, doc in enumerate(read_xlsx('sna_v2.xlsx', 'rawdata')):
        # doc_splitted = doc.split()
        # print(doc)
        doc_splitted = map(lambda x: x.upper(), doc.split())
        for token in doc_splitted:
            # print(token)
            freq[token] += 1
        if i % 10000 == 0:
            with open(freq_word_path, 'wb') as freq_word_f:
                pickle.dump(freq, freq_word_f)
            print('dump at {}'.format(i))
    with open(freq_word_path, 'wb') as freq_word_f:
        pickle.dump(freq, freq_word_f)
    print('dump completed at {}'.format(i))
    docs_cnt = i
    return docs_cnt, freq


def word_doc_freq():
    freq_doc_path = os.path.join(data_dir, doc_freq_fn)
    try:
        with open(freq_doc_path, 'rb') as freq_doc_f:
            freq_doc = pickle.load(freq_doc_f)
            print('word-doc frequency distribution loaded')
        docs_cnt = len(freq_doc)
        return docs_cnt, freq_doc
    except IOError:
        pass

    print('counting document frequency')
    freq_doc = defaultdict(int)
    for i, doc in enumerate(read_xlsx('sna_v2.xlsx', 'rawdata')):
        doc_splitted = map(lambda x: x.upper(), doc.split())
        for token in set(doc_splitted):
            # print(token)
            freq_doc[token] += 1
        if i % 10000 == 0:
            with open(freq_doc_path, 'wb') as freq_doc_f:
                pickle.dump(freq_doc, freq_doc_f)
            print('dump at {}'.format(i))
    with open(freq_doc_path, 'wb') as freq_doc_f:
        pickle.dump(freq_doc, freq_doc_f)
    print('dump completed at {}'.format(i))
    docs_cnt = i
    # print(freq_doc)
    return docs_cnt, freq_doc


def build_vocabulary(lower=3, n=50000):
    path = os.path.join(data_dir, vocab_fn)
    try:
        with open(path, 'rb') as vocab_f:
            vocab_temp = pickle.load(vocab_f)
            print('vocabulary loaded')
            print(vocab_temp)
            return vocab_temp
    except IOError:
        print('fail to load vocab')
        pass

    print('building vocabulary')
    # freq = build_word_frequency_distribution()
    # print('freq: ', freq)
    top_words = list(sorted(word_freq.items(), key=lambda x: -x[1]))[:n-lower+1]
    vocab_temp = {}
    i = lower
    for w, freq in top_words:
        vocab_temp[w] = i
        i += 1
    with open(path, 'wb') as vocab_file:
        pickle.dump(vocab_temp, vocab_file)
    print('creating vocab completed: ', vocab_temp)
    return vocab_temp


def idf(doc_cnt, doc_freq):
    idf_temp = []
    for key, value in doc_freq.items():
        # print(key, value)
        idf_temp.append([key, log(doc_cnt / (value + 1))])
    return idf_temp


doc_cnt, word_freq = word_freq()
doc_cnt2, doc_freq = word_doc_freq()

print(doc_cnt, doc_cnt2)
vocab = build_vocabulary(lower=3, n=50000)
print(len(vocab))
# vocab.sort()

print('length of word frequency: ', len(word_freq))     # 16545
print(word_freq['NICKEL'])  # 2013 -> 2642(with upper case)
print(doc_freq['NICKEL'])   # 1778
"""
dump completed at 17093
length of word frequency:  16545
"""

word_idf = idf(doc_cnt, doc_freq)
# print(word_idf[:10])
word_idf = np.array(word_idf)

alpha = 0.9

df1 = pd.DataFrame(word_idf[:, 1], columns=['word_idf'], index=word_idf[:, 0])
df2 = df1.astype(float)
cutoff = df2['word_idf'].quantile(q=0.25)
vocab_df = df2[df2['word_idf'] >= cutoff * alpha]
# print('Alpha= ', alpha)
# print(cutoff * alpha, vocab_df.head(10))

# print(vocab_df.index)
vocab_list = list(vocab_df.index)
print(vocab_list[:10])

# for i in range(5, 11):
#     vocab_df = df2[df2['word_idf'] >= cutoff * i/10]
#     print('Alpha= ', i/10, ', cutoff= ', cutoff * i/10, ', count= ', vocab_df.count())
#     # print(vocab_df.describe())
#     print(vocab_df.head(10))


def wordcloud_applicant():
    applicant = ['GE', 'SUMITOMO', 'HITACHI', 'NIPPON STEEL', 'SIEMENS',
                 'MITSUBISHI', 'UNITED TECHNOLOGIES', 'TOSHIBA', 'JXTG GROUP', 'NGK SPARK PLUG']

    df_cloud = pd.read_excel('sna_v2.xlsx', sheet_name='rawdata')
    df_cloud = df_cloud[df_cloud['발행국'] == 'USPTO']
    for i, app in enumerate(applicant):
        df_tmp = df_cloud[df_cloud['출원인/특허권자'] == app]
        df_tmp = list(df_tmp['발명의명칭'])
        temp = []
        for doc in df_tmp:
            doc_splitted = [token.upper() for token in doc.split() if token.upper() in vocab_list]
            temp.extend(doc_splitted)
        wc1 = WordCloud(max_font_size=200, stopwords=stop_set, background_color='white',
                        max_words=100,
                        # random_state=42,
                        width=800, height=800)
        wc1.generate(' '.join(temp))
        plt.figure(figsize=(10, 8))
        plt.imshow(wc1)
        plt.tight_layout(pad=0)
        plt.axis('off')
        # plt.show()
        plt.savefig(os.path.join(img_dir, str(i)+'. '+app+'.png'))


# wordcloud_applicant()


def sna_applicant():
    applicant = ['GE', 'SUMITOMO', 'HITACHI', 'NIPPON STEEL', 'SIEMENS',
                 'MITSUBISHI', 'UNITED TECHNOLOGIES', 'TOSHIBA', 'JXTG GROUP', 'NGK SPARK PLUG',
                 'JINCHUAN', 'CAS', 'ATI PROPERTIES', 'SICHUAN NORMAL UNIVERSITY', 'PANASONIC',
                 'JFE', 'CENTRAL SOUTH UNIV', 'ROLLS-ROYCE', 'TDK', 'NATIONAL INSTITUTE FOR MATERIALS SCIENCE',
                 'KOBE STEEL', 'HONEYWELL', 'POSCO', 'IHI', 'SANDVIK']
    # JINCHUAN, SICHUAN NORMAL UNIVERSITY는 단어가 추출되지 않음
    sna_text = []
    df_cloud = pd.read_excel('sna_v2.xlsx', sheet_name='rawdata')
    df_cloud = df_cloud[df_cloud['발행국'] == 'USPTO']
    for i, app in enumerate(applicant):
        df_tmp = df_cloud[df_cloud['출원인/특허권자'] == app]
        df_tmp = list(df_tmp['발명의명칭'])
        temp = []
        # for doc in df_tmp:
        #     doc_splitted = [token.upper() for token in doc.split() if token.upper() in vocab_list]
        #     temp.extend(doc_splitted)
        # sna_text.append([app, temp])
        for doc in df_tmp:
            for token in doc.split():
                if token.upper() in vocab_list:
                    sna_text.append([app, token.upper()])
    return sna_text
# vocab list와 비교했더니, 2개의 출원인은 단어가 추출되지 않음
# 각 출원인마다 cutoff를 서로 다르게 적용해야 할까?

# term-doc 매트릭스를 만들고, LSA를 실행하자
# 특허의 유사도를 구하고, 두 출원인의 특허간 유사도 평균을 edge로 사용해보자


sna_text_cleaned = np.array(sna_applicant())
# print(sna_text_cleaned[:3])
sna_df = pd.DataFrame({'App': sna_text_cleaned[:, 0], 'Word': sna_text_cleaned[:, 1]})
sna_df.to_csv(os.path.join(data_dir, 'sna_text_cleaned.csv'),
              encoding='utf-8', sep='\t')
# with open(os.path.join(data_dir, 'sna_text_cleaned.csv'), 'w', encoding='utf8', newline='') as csv_f:
#     for row in sna_text_cleaned:
#         csv_f.writelines(row)
# # csv_f.close()
# print('csv file saving completed.')

# def make_node(text):
#     result = []
#     return result
#
#
# def make_edge():
#     result = []
#     return result


# df_cloud = pd.read_excel('sna_v2.xlsx', sheet_name='rawdata')
# # print(df_cloud.count())
# df_cloud = df_cloud[df_cloud['발행국'] == 'USPTO']
# # print(df_cloud.count())
# # df_cloud = df_cloud[df_cloud['출원인/특허권자'] == 'GE']
# df_cloud = df_cloud[df_cloud['출원인/특허권자'] == 'SUMITOMO']
#
# # print(df_cloud.count())
# df_cloud = list(df_cloud['발명의명칭'])
# print('df_cloud: ', df_cloud[:10])
#
# temp = []
# for doc in df_cloud:
#     # doc_splitted = map(lambda x: x.upper(), doc.split())
#     print(doc)
#     # temp.append([token for token in doc_splitted if token in vocab_list])
#     doc_splitted = [token.upper() for token in doc.split() if token.upper() in vocab_list]
#     print(doc_splitted)
#     temp.extend(doc_splitted)
#
# # temp = ' '.join(temp)
# print(temp)
# wc1 = WordCloud(max_font_size=200, stopwords=stop_set, background_color='white',
#                 max_words=200, random_state=42,
#                 width=800, height=800)
# wc1.generate(' '.join(temp))
# plt.figure(figsize=(10, 8))
# plt.imshow(wc1)
# plt.tight_layout(pad=0)
# plt.axis('off')
# plt.show()
# plt.savefig(os.path.join(img_dir, 'idf_hist.png'))


# plot_hist(vocab_df['word_idf'])

# print(df.shape, df.head())     # (16545, 2)

# # tf_idf(df['발명의명칭'].tolist()[:20])
#
# # print(len(freq))    # 16545
# # print(freq.most_common(5))
# # [('of', 5474), ('and', 4847), ('method', 3782), ('및', 3068), ('for', 2788)]
# # 위 숫자는 빈도수
# # 문서수 역시 중요하므로 추출 필요
#
# # make_edge()
#
# # node : Id, Label, attributes
# # edge : Source, Target, Weight,
#
# # 시트명 : rawdata
# # 기술분야	Publication Number-발행번호	발행일	문서코드	Application Number-출원번호	출원일	출원년
# # 출원년구간(4년단위)	출원년구간(10年)	출원년구간(성장단계용)	발행국4	발행국	INPADOC Family ID
# # 출원인	출원인국가	출처	출원인(정비)	출원인국가(정비)	출원인_구분
# # 출원인/특허권자	출원인 국적	출워인 형태	출원인 국적(IP5&기타)
# # 내/외국인	내국인	외국인	발명의명칭	특허번호(등록번호)	특허권 발생일(등록일)
# # IPC(클래스)	IPC	KR진입	EP진입	US진입	JP진입	CN진입
# # IP5진입수	2극이상	3극이상	착수여부	최종처분일	등록일자	등록여부	심사결과
# # 등록년구간-심사결과(4年)	등록년구간-심사결과(10年)	2극이상등록결정
# # TOP5(글로벌TOP5/국내TOP5)	TOP5(2극이상출원기준)
# # 법적코드	법적설명	법적상세

