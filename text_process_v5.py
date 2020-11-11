from utils import *
from text_utils import *
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import numpy as np
import nltk
import re
import os
# nltk.download('wordnet')
from PIL import Image


# https://igor.mp/blog/2019/12/31/tfidf-python.html 참고

starting_time = time.time()
set_pandas_display_options()

df_excel = pd.read_excel('IOT_1029_1.xlsx')
df_excel = df_excel.set_index('키위번호')
print(proc_time(starting_time), ' 1. Load Excel file with', len(df_excel), 'patents')  # Success to load Excel file with 17094 patents

print(df_excel.columns)     # 컬럼 구성 확인
print(df_excel.head(1))

df_kor = df_excel[df_excel['발행국'].isin(['KIPO', 'JPO', 'KR', 'JP'])]
print(proc_time(starting_time), ' 2. Identify KR, JP', len(df_kor), 'patents')
df_eng = pd.DataFrame(df_excel[df_excel['발행국'].isin(['USPTO', 'EPO', 'CNIPA', 'US', 'EP', 'CN'])])
print(proc_time(starting_time), ' 3. Identify US, EP, CN', len(df_eng), 'patents')

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
    print(df_eng[['발명의명칭', 'title_token', 'title_pos', 'title_lemma', 'title_stem']].head())

# 문자열 변화 비교
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

tf_dict = {}
df_dict = {}
idf_dict = {}

# docs = df_eng['title_stem'].head().tolist()
docs = df_eng['title_lemma'].tolist()
doc_cnt = len(docs)

# print(docs)
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

# print('tf_dict: ', tf_dict)
# print('df_dict: ', df_dict)
print('idf_dict: ', idf_dict)
# idf_df = pd.DataFrame.from_dict(data=idf_dict, orient=idf_dict.keys())
idf_df = pd.Series(idf_dict).to_frame()
print(idf_df.sort_values(by=0, ascending=True).head(10))
print(idf_df.sort_values(by=0, ascending=False).head(10))

# mask_im = np.array(Image.open('mask.png'))
# wc = WordCloud(max_font_size=150, stopwords=stopwords.words("english"), mask=mask_im, background_color='white',
#                max_words=int(1000), random_state=42, width=1024, height=768)
#
# wc = wc.generate_from_frequencies(idf_dict)
# wc.to_file(os.path.join('img', "word_cloud_lemma_idf.png"))
