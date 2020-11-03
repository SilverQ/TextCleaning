import pandas as pd
from math import log
import os
from collections import Counter, defaultdict
import pickle


# # for loop 1'st
# for text in df_kor['발명의명칭']:
#     print('Original Text : ', text)
#     print('Tokenized Text : ', tokenizer.tokenize(text), '\n')

# # for loop 2'nd, https://3months.tistory.com/419, iterrow에 비해 3배 빠름
# for i in df_kor.index:
#     text = df.loc[i, '발명의명칭']
#     print(text, tokenizer.tokenize(text))

# # 이제 없어지는듯. underscore를 사용해야 객체 사용이 가능한데, protected member라고 나옴
# # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Index.get_value.html?highlight=get_value
# # 여기엔 Fast lookup of value from 1-dimensional ndarray., Only use this if you know what you’re doing. 라고 나옴
# for i in df_kor.index:
#     text = df_kor._get_value(i, '발명의명칭')
#     print(text, tokenizer.tokenize(text))


def test_excel():
    df = pd.read_excel('sna_v2.xlsx', sheet_name='rawdata')
    # 17,000건의 Ni합금
    print(df.head())
    print(df['Application Number-출원번호'].head())
    print(df.loc[1:3, ['Application Number-출원번호', '발명의명칭']])
    print(df.columns)
    # Application Number-출원번호, 발명의명칭


def test_excel2():
    df = pd.read_excel('sna_v2.xlsx', sheet_name='rawdata')
    docs = df['발명의명칭'].tolist()
    doc = docs[0]
    print(doc)
    print(doc.count("nitinol"))


def stat(df):
    df_min = df.min()
    df_max = df.max()
    df_mean = df.mean()
    df_quantile = df.quantile(q=0.25)
    df_sum = df.sum()
    df_median = df.median()

    print('Minimum of idf: ', df_min)
    print('Maximum of idf: ', df_max)
    print('Mean of idf: ', df_mean)
    print('Quantile of idf: ', df_quantile)
    print('Sum of idf: ', df_sum)
    print('Median of idf: ', df_median)


# stat(df['word_idf'])
# print(df['word_idf'].mean())
"""
    Minimum of idf:  1.373860648365285
    Maximum of idf:  9.020692039795767
    Mean of idf:  8.474060550535837
    Sum of idf:  140203.33180861542
    Median of idf:  9.020692039795767
"""


# wikidocs의 tf-idf, 하지만 대량의 특허 데이터를 한번에 처리 불가능
# 나는 전체 입력에 대한 idf를 연산해서, 불용어 처리에 사용할 예정
def tf_idf(docs):
    # https: // wikidocs.net / 31698
    splitted_input = list(set(w for doc in docs for w in doc.split(' ')))
    print('len of sentence: ', len(docs), ', cnt of words: ', len(splitted_input))
    print('sample of words: ', splitted_input[:10])
    # 전체 docs를 일괄입력했음. 전체 데이터로 연산하려면 generator 객체로 돌리고, 중간중간 저장 필요
    # 과거에 수행했던 계층별 주의 네트워크 결과물을 사용해보자
    # splited_input.sort()
    doc_cnt = len(docs)

    def tf(word, doc):
        return doc.count(word)

    def idf(word):
        df_temp = 0
        for doc in docs:
            df_temp += word in doc
        return log(doc_cnt / (df_temp + 1))

    def tfidf(word, doc):
        return tf(word, doc) * idf(word)

    result = []
    for i in range(len(docs)):  # 각 문서에 대해서 아래 명령을 수행
        result.append([])
        d = docs[i]
        for j in range(len(splitted_input)):
            t = splitted_input[j]
            result[-1].append(tf(t, d))
    # 모든 문서에 대해 사전의 모든 단어를 대입하여 건수 확인... 매우 느릴듯

    tf_ = pd.DataFrame(result, columns=splitted_input)
    print('shape of tf: ', tf_.shape, tf_.head())   # (10, 95)
    '''
    print(result[:20])
    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 10, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]]
    '''

    result = []
    for j in range(len(splitted_input)):
        t = splitted_input[j]
        result.append(idf(t))

    idf_ = pd.DataFrame(result, index=splitted_input, columns=["IDF"])
    print('shape of idf: ', idf_.shape, idf_.head(20))    # (95, 1)

    result = []
    for i in range(len(docs)):  # 각 문서에 대해서 아래 명령을 수행
        result.append([])
        d = docs[i]
        for j in range(len(splitted_input)):
            t = splitted_input[j]
            result[-1].append(tfidf(t, d))
    tf_idf_ = pd.DataFrame(result, columns=splitted_input)
    print(tf_idf_)
    return tf_idf_


def tf_idf_v2(docs):
    splitted_input = list(set(w for doc in docs for w in doc.split(' ')))
    print('len of sentence: ', len(docs), ', cnt of words: ', len(splitted_input))
    print('sample of words: ', splitted_input[:10])
    # 전체 docs를 일괄입력했음. 전체 데이터로 연산하려면 generator 객체로 돌리고, 중간중간 저장 필요
    # 과거에 수행했던 계층별 주의 네트워크 결과물을 사용해보자
    # splited_input.sort()
    doc_cnt = len(docs)

    def tf(word, doc):
        return doc.count(word)

    def idf(word):
        df_temp = 0
        for doc in docs:
            df_temp += word in doc
        return log(doc_cnt / (df_temp + 1))

    def tfidf(word, doc):
        return tf(word, doc) * idf(word)

    result = []
    for i in range(len(docs)):  # 각 문서에 대해서 아래 명령을 수행
        result.append([])
        d = docs[i]
        for j in range(len(splitted_input)):
            t = splitted_input[j]
            result[-1].append(tf(t, d))
    # 모든 문서에 대해 사전의 모든 단어를 대입하여 건수 확인... 매우 느릴듯

    tf_ = pd.DataFrame(result, columns=splitted_input)
    print('shape of tf: ', tf_.shape, tf_.head())   # (10, 95)
    '''
    print(result[:20])
    [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0, 10, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]]
    '''

    result = []
    for j in range(len(splitted_input)):
        t = splitted_input[j]
        result.append(idf(t))

    idf_ = pd.DataFrame(result, index=splitted_input, columns=["IDF"])
    print('shape of idf: ', idf_.shape, idf_.head(20))    # (95, 1)

    result = []
    for i in range(len(docs)):  # 각 문서에 대해서 아래 명령을 수행
        result.append([])
        d = docs[i]
        for j in range(len(splitted_input)):
            t = splitted_input[j]
            result[-1].append(tfidf(t, d))
    tf_idf_ = pd.DataFrame(result, columns=splitted_input)
    print(tf_idf_)


# Method 1
def to_df1(np_array):
    print('to_df1')
    df1 = pd.DataFrame(np_array[:, 1], columns=['word_idf'], index=np_array[:, 0])
    print(df1.describe())
    df2 = df1.astype(float)
    print(df2.describe())
    return df2


# Method 2
def to_df2(np_array):
    print('to_df2')
    df = pd.DataFrame(np_array[:, 0], columns=['word'])
    df['word_idf'] = np_array[:, 1]
    print(df.describe())
    # print(df.head(5))
    # print(np.mean(word_idf[:, 1]))
    # print(df.head(5))


def build_word_frequency_distribution():
    freq_word_path = os.path.join(data_dir, word_freq_fn)
    freq_doc_path = os.path.join(data_dir, doc_freq_fn)
    try:
        with open(freq_word_path, 'rb') as freq_word_f:
            freq_word = pickle.load(freq_word_f)
        with open(freq_doc_path, 'rb') as freq_doc_f:
            freq_doc = pickle.load(freq_doc_f)
            print('frequency distribution loaded')
        docs_cnt = len(freq_word)
        return docs_cnt, freq_word, freq_doc
    except IOError:
        pass

    print('building frequency distribution')
    freq = defaultdict(int)
    freq_doc = defaultdict(int)
    for i, doc in enumerate(read_xlsx('sna_v2.xlsx', 'rawdata')):
        # doc_splitted = doc.split()
        doc_splitted = map(lambda x: x.upper(), doc.split())
        for token in doc_splitted:
            # print(token)
            freq[token] += 1
        for token in set(doc_splitted):
            freq_doc[token] += 1
        if i % 10000 == 0:
            with open(freq_word_path, 'wb') as freq_word_f:
                pickle.dump(freq, freq_word_f)
            with open(freq_doc_path, 'wb') as freq_doc_f:
                pickle.dump(freq_doc, freq_doc_f)
            print('dump at {}'.format(i))
    with open(freq_word_path, 'wb') as freq_word_f:
        pickle.dump(freq, freq_word_f)
    with open(freq_doc_path, 'wb') as freq_doc_f:
        pickle.dump(freq_doc, freq_doc_f)
    print('dump completed at {}'.format(i))
    docs_cnt = i
    return docs_cnt, freq, freq_doc


text = [('EXTREMELY', 'NNP'), ('FINE', 'NNP'), ('SHAPE', 'NNP'), ('MEMORY', 'NNP'), (',', ',')]
print([word[0]for word in text if word[1] not in [',']])
