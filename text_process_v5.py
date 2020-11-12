from utils import *
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import os
from PIL import Image
# from text_utils import *
# import nltk
# import re
# from collections import Counter
# nltk.download('wordnet')

# https://igor.mp/blog/2019/12/31/tfidf-python.html 참고

set_pandas_display_options()


def load_file(fname):
    if os.path.exists(fname[:-5]+'.pickle'):
        print('전처리 결과가 존재합니다, 이전 처리 결과를 활용합니다')
        df_eng = pd.read_pickle(fname[:-5]+'.pickle')
        print_step(1, 'Load preprocessed file')
        return df_eng
    elif fname[-4:] == 'xlsx':
        df_excel = pd.read_excel(fname)
        df_excel = df_excel.set_index('키위번호')
        print_step(1-1, 'Load Excel file with ' + str(len(df_excel)) + ' patents')
        df_kor = df_excel[df_excel['발행국'].isin(['KIPO', 'JPO', 'KR', 'JP'])]
        df_eng = pd.DataFrame(df_excel[df_excel['발행국'].isin(['USPTO', 'EPO', 'CNIPA', 'US', 'EP', 'CN'])])
        print_step(1-2, 'Split Excel file with Language')
        # 분석 범위 한정
        nat = 'US'
        df_eng = df_eng[df_eng['발행국'].isin(['USPTO', 'US'])]
        lemmatizer = WordNetLemmatizer()
        stemmer = SnowballStemmer('english')
        stop_pos_list = stop_pos(1)  # 1~3으로 입력해서 사용하자, 1은 전치사 등 최소 제거, 2는 부사형 제거, 3은 동사형 제거까지

        if '발명의명칭' in df_eng.columns:
            title = df_eng['발명의명칭']
            df_eng['title_token'] = df_eng['발명의명칭'].apply(lambda x: word_tokenize(x))
            df_eng['title_pos'] = df_eng['title_token'].apply(lambda x: pos_tag(x))
            df_eng['title_lemma'] = df_eng['title_pos'].apply(
                lambda x: [lemmatizer.lemmatize(word[0].lower(), get_wordnet_pos(word[1]))
                           for word in x
                           # if word[1] not in stop_pos_list
                           ]
            )
            df_eng['title_stem'] = df_eng['title_lemma'].apply(
                lambda x: [stemmer.stem(word) for word in x]
            )

        print_step(1-3, 'Language preprocess done(Lemmatize, POS-TAG & Stem)')
        print(df_eng[['발명의명칭', 'title_token', 'title_pos', 'title_lemma', 'title_stem']].head(2))
        df_eng.to_pickle(fname[:-5]+'.pickle')
        return df_eng
    else:
        return None


def tf_idf(docs):
    tf_dict = {}
    df_dict = {}
    idf_dict = {}
    for doc in docs:
        # print(doc, set(doc))
        for word in set(doc):
            # print(word, sum([1 for w in doc if w == word]))
            if word in tf_dict.keys():
                tf_dict[word] += sum([1 for w in doc if w == word])
                df_dict[word] += 1
            else:
                tf_dict[word] = sum([1 for w in doc if w == word])
                df_dict[word] = 1
    for word in df_dict.keys():
        idf_dict[word] = np.log(doc_cnt / (df_dict[word] + 1))
    return tf_dict, df_dict, idf_dict


def draw_cloud(data, idf_rank=0):
    # print(data.describe())
    if idf_rank > 0:
        word_cloud_df = data[data['idf_rank'] >= idf_rank]
        word_cloud_dict = word_cloud_df['term_freq'].to_dict()
        # print(word_cloud_dict)

        mask_im = np.array(Image.open('mask1.png'))
        wc = WordCloud(max_font_size=150, stopwords=stopwords.words("english"), mask=mask_im, background_color='white',
                       max_words=int(1000), random_state=42, width=1024, height=768)

        wc = wc.generate_from_frequencies(word_cloud_dict)
        wc.to_file(os.path.join('img', 'word_cloud_idf_rank under'+str(idf_rank)+'.png'))


# 1단계: 파일 입력(xlsx만 입력 가능)
f_name = input('사용할 파일의 명칭을 입력하세요\n파일명: ')
print('입력한 파일 명은 ', f_name[:-5], ', 확장자는 ', f_name[-4:], '입니다.')
df_eng = load_file(f_name)

# 2단계: term frequency 산출
# title_list = df_eng['title_lemma'].tolist()
doc_cnt = len(df_eng['title_lemma'].tolist())
term_freq, doc_freq, inverse_df = tf_idf(df_eng['title_lemma'].tolist())
print_step(2, 'Calculate TF & IDF')

sample_word = 'mobile'
print('Sample Word: [', sample_word, '], term_freq: ', term_freq[sample_word],
      ', doc_freq: ', doc_freq[sample_word], ', inverse_df: ', inverse_df[sample_word])

tf_idf_df = pd.DataFrame({'term_freq': pd.Series(term_freq),
                          'doc_freq': pd.Series(doc_freq),
                          'idf': pd.Series(inverse_df)})
tf_idf_df['idf_rank'] = tf_idf_df['idf'].rank()
print_step(3, 'Calculate IDF Rank')
print(tf_idf_df.sort_values(by='idf', ascending=True).head(3))

draw_cloud(data=tf_idf_df, idf_rank=5)
print_step(4, 'Finished to save Wordcloud')

# for i in range(0, 101, 10):
#     draw_cloud(data=tf_idf_df, idf_rank=i)

# 조건 : lemmatize only, stem, idf rank, pos tag
# 분석 : 국가별, 출원인별
# Flow
# 1. 입력할 파일 선택 (발행번호 목록? CMS 데이터? - 텍스트 포함여부)
#  - 동일한 이름의 '.pickle' 파일에 텍스트 처리 결과를 임시 저장하고, 해당 파일이 있으면 재활용
# 2. 분석의 범위 선택 (발명의 명칭, 요약, 청구항, 발행국가별, 출원인국가별, 구간별, 출원인별)
# 3. 데이터 추출 (CMS는 열기, 발행번호는 DB에서 취득)
# 4. 데이터 클린징 (발명의 명칭, 요약?)
# 5. 워드 클라우드 생성 및 옵션 조정 (idf rank, 형태소 목록)

"""
tokens:  ['Interactive', 'ID', 'system', 'using', 'mobile', 'devices']
lemmatized_tokens:  ['interactive', 'id', 'system', 'using', 'mobile', 'device']
stem_tokens:  ['interact', 'id', 'system', 'use', 'mobil', 'devic']
tokens:  ['Method', 'for', 'monitoring', 'by', 'collaborating', 'between', 'MTC', 'devices', ',', 'related', 'device', 'and', 'system']
lemmatized_tokens:  ['method', 'for', 'monitoring', 'by', 'collaborating', 'between', 'mtc', 'device', ',', 'related', 'device', 'and', 'system']
stem_tokens:  ['method', 'for', 'monitor', 'by', 'collabor', 'between', 'mtc', 'devic', ',', 'relat', 'devic', 'and', 'system']
tokens:  ['System', 'and', 'method', 'for', 'connecting', ',', 'configuring', 'and', 'testing', 'new', 'wireless', 'devices', 'and', 'applications']
lemmatized_tokens:  ['system', 'and', 'method', 'for', 'connecting', ',', 'configuring', 'and', 'testing', 'new', 'wireless', 'device', 'and', 'application']
stem_tokens:  ['system', 'and', 'method', 'for', 'connect', ',', 'configur', 'and', 'test', 'new', 'wireless', 'devic', 'and', 'applic']
tokens:  ['Detection', 'of', 'stale', 'encryption', 'policy', 'by', 'group', 'members']
lemmatized_tokens:  ['detection', 'of', 'stale', 'encryption', 'policy', 'by', 'group', 'member']
stem_tokens:  ['detect', 'of', 'stale', 'encrypt', 'polici', 'by', 'group', 'member']
tokens:  ['INTEGRATED', 'PHYSICAL', 'AND', 'LOGICAL', 'SECURITY', 'MANAGEMENT', 'VIA', 'A', 'PORTABLE', 'DEVICE']
lemmatized_tokens:  ['integrated', 'physical', 'and', 'logical', 'security', 'management', 'via', 'a', 'portable', 'device']
stem_tokens:  ['integr', 'physic', 'and', 'logic', 'secur', 'manag', 'via', 'a', 'portabl', 'devic']

키위번호
Interactive ID system using mobile devices
[Interactive, ID, system, using, mobile, devices]
[(Interactive, NNP), (ID, NNP), (system, NN), (using, VBG), (mobile, JJ), (devices, NNS)]
[interactive, id, system, use, mobile, device]
[interact, id, system, use, mobil, devic]

Method for monitoring by collaborating between MTC devices, related device and system
[Method, for, monitoring, by, collaborating, between, MTC, devices, ,, related, device, and, system]
[(Method, NNP), (for, IN), (monitoring, NN), (by, IN), (collaborating, VBG), (between, IN), (MTC, NNP), (devices, NNS), (,, ,), (related, JJ), (device, NN), (and, CC), (system, NN)]
[method, monitoring, collaborate, mtc, device, related, device, system]
[method, monitor, collabor, mtc, devic, relat, devic, system]

System and method for connecting, configuring and testing new wireless devices and applications
[System, and, method, for, connecting, ,, configuring, and, testing, new, wireless, devices, and, applications]
[(System, NN), (and, CC), (method, NN), (for, IN), (connecting, VBG), (,, ,), (configuring, VBG), (and, CC), (testing, VBG), (new, JJ), (wireless, JJ), (devices, NNS), (and, CC), (applications, N...
[system, method, connect, configure, test, new, wireless, device, application]
[system, method, connect, configur, test, new, wireless, devic, applic]

Detection of stale encryption policy by group members
[Detection, of, stale, encryption, policy, by, group, members]
[(Detection, NN), (of, IN), (stale, JJ), (encryption, NN), (policy, NN), (by, IN), (group, NN), (members, NNS)]
[detection, stale, encryption, policy, group, member]
[detect, stale, encrypt, polici, group, member]

INTEGRATED PHYSICAL AND LOGICAL SECURITY MANAGEMENT VIA A PORTABLE DEVICE
[INTEGRATED, PHYSICAL, AND, LOGICAL, SECURITY, MANAGEMENT, VIA, A, PORTABLE, DEVICE]
[(INTEGRATED, NNP), (PHYSICAL, NNP), (AND, NNP), (LOGICAL, NNP), (SECURITY, NNP), (MANAGEMENT, NNP), (VIA, NNP), (A, NNP), (PORTABLE, NNP), (DEVICE, NNP)]
[integrated, physical, and, logical, security, management, via, a, portable, device]
[integr, physic, and, logic, secur, manag, via, a, portabl, devic]
"""

