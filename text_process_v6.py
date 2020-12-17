from utils import *
import sys
import time
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tag.perceptron import PerceptronTagger
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import os
from PIL import Image
# from nltk.tag import pos_tag
# import seaborn as sns
# from text_utils import *
# import nltk
# import re
# from collections import Counter
# nltk.download('wordnet')

# https://igor.mp/blog/2019/12/31/tfidf-python.html 참고

set_pandas_display_options()
lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')
# tagger = pos_tag()
tagger = PerceptronTagger()


@Timer("load_file()")
def load_file(fname):
    if os.path.exists(fname[:-5]+'.pickle'):
        print('전처리 결과가 존재합니다, 이전 처리 결과를 활용합니다')
        df_eng = pd.read_pickle(fname[:-5]+'.pickle')
        print_step(1, 'Load preprocessed file')
        return df_eng
    elif fname[-4:] == 'xlsx':
        df_excel = pd.read_excel(fname)
        df_excel = df_excel.set_index('Publication Number-발행번호')
        print_step(1-1, 'Load Excel file with ' + str(len(df_excel)) + ' patents')
        df_kor = df_excel[df_excel['발행국'].isin(['KIPO', 'JPO', 'KR', 'JP'])]
        df_eng = pd.DataFrame(df_excel[df_excel['발행국'].isin(['USPTO', 'EPO', 'CNIPA', 'US', 'EP', 'CN'])])
        print_step('1-2', 'Split Excel file with Language')
        # 분석 범위 한정
        nat = 'US'
        # df_eng = df_eng[df_eng['발행국'].isin(['USPTO', 'US'])]
        stop_pos_list = stop_pos(1)  # 1~3으로 입력해서 사용하자, 1은 전치사 등 최소 제거, 2는 부사형 제거, 3은 동사형 제거까지

        if '발명의명칭' in df_eng.columns:
            title = df_eng['발명의명칭']
            df_eng['title_token'] = df_eng['발명의명칭'].apply(lambda x: word_tokenize(x))
            # df_eng['title_pos'] = df_eng['title_token'].apply(lambda x: pos_tag(x))
            df_eng['title_pos'] = df_eng['title_token'].apply(lambda x: tagger.tag(x))
            df_eng['title_lemma'] = df_eng['title_pos'].apply(
                lambda x: [lemmatizer.lemmatize(word[0].lower(), get_wordnet_pos(word[1]))
                           for word in x
                           # if word[1] not in stop_pos_list
                           ]
            )
            df_eng['title_lemma1'] = df_eng['title_pos'].apply(
                lambda x: [(lemmatizer.lemmatize(word[0].lower(), get_wordnet_pos(word[1])),
                            word[1])
                           for word in x
                           # if word[1] not in stop_pos_list
                           ]
            )
            df_eng['title_stem'] = df_eng['title_lemma'].apply(
                lambda x: [stemmer.stem(word) for word in x]
            )
            df_eng['title_stem1'] = df_eng['title_lemma1'].apply(
                lambda x: [(stemmer.stem(word[0]), word[1]) for word in x]
            )

        print_step('1-3', 'Language preprocess done(Lemmatize, POS-TAG & Stem)')
        print(df_eng[['발명의명칭', 'title_token', 'title_pos', 'title_lemma', 'title_stem']].head(2))
        df_eng.to_pickle(fname[:-5]+'.pickle')
        return df_eng
    else:
        return None


# @Timer("tf_idf()")
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


@Timer("get_major_pos()")
def get_major_pos(docs):
    # major_pos = {}
    w_list = []
    p_list = []
    for sen in docs:
        for word in sen:
            w_list.append(word[0])
            p_list.append(word[1])
    wnp_df = pd.DataFrame({'w': np.array(w_list), 'p': np.array(p_list)})
    wnp_df['cnt'] = 1
    wnp_df1 = wnp_df.groupby(['w', 'p']).sum().reset_index()
    # print(wnp_df1.head(3))
    wnp_df1['row_num'] = wnp_df1.sort_values(['w', 'cnt'], ascending=[True, False]).groupby(['w']).cumcount() + 1
    # print(wnp_df1.head(30))
    # print(wnp_df2[wnp_df2['w'] == 'a'])
    wnp_df1 = wnp_df1[wnp_df1['row_num'] == 1]
    wnp_df1 = wnp_df1.set_index('w')
    wnp_df1 = wnp_df1[['p']]
    # print(wnp_df1.head(3))
    result = wnp_df1.to_dict('series')['p']
    # print(result)
    return result


# @Timer("draw_cloud()")
def draw_cloud(data):
    mask_im = np.array(Image.open('mask1.png'))
    wc = WordCloud(max_font_size=150, stopwords=stopwords.words("english"), mask=mask_im, background_color='white',
                   max_words=int(1000), random_state=42, width=1600, height=960)

    wc = wc.generate_from_frequencies(data)
    # wc.to_file(os.path.join('img', 'word_cloud_idf_rank under'+str(idf_rank)+'.png'))
    return wc


# 1단계: 파일 입력(xlsx만 입력 가능)
# f_name = input('사용할 파일의 명칭을 입력하세요\n파일명: ')
f_name = sys.argv[-1]   # .py 실행 시 argument 입력
print(f_name)
if len(f_name) == 0:
    print('No file name')
else:
    print('입력한 파일 명은 ', f_name[:-5], ', 확장자는 ', f_name[-4:], '입니다.')
    df_eng = load_file(f_name)
    print(df_eng.head(2))

    # 2단계: term frequency 산출
    # title_list = df_eng['title_lemma'].tolist()
    doc_cnt = len(df_eng['title_lemma'].tolist())
    term_freq, doc_freq, inverse_df = tf_idf(df_eng['title_lemma'].tolist())
    major_pos = get_major_pos(df_eng['title_lemma1'].tolist())
    # sample_word = 'mobile'
    # print('Sample Word: [', sample_word, '], term_freq: ', term_freq[sample_word],
    #       ', doc_freq: ', doc_freq[sample_word], ', inverse_df: ', inverse_df[sample_word])
    print_step(2, 'Calculate TF & IDF')
    tf_idf_df = pd.DataFrame({'term_freq': pd.Series(term_freq),
                              'doc_freq': pd.Series(doc_freq),
                              'idf': pd.Series(inverse_df),
                              'major_pos': pd.Series(major_pos)})
    tf_idf_df['idf_rank'] = tf_idf_df['idf'].rank()
    print_step(3, 'Calculate IDF Rank')
    print(tf_idf_df.sort_values(by='idf', ascending=True).head(2))

    # 조건 필터: idf 순위, pos tag
    stop_pos_list = stop_pos(3)  # 1~3으로 입력해서 사용하자, 1은 전치사 등 최소 제거, 2는 부사형 제거, 3은 동사형 제거까지
    print('stop_pos_list: ', stop_pos_list)
    idf_rank = 10
    # print(idf_rank)
    data = tf_idf_df[tf_idf_df['idf_rank'] >= idf_rank]
    # data_us = df_eng['title_lemma']
    data_us = pd.DataFrame(df_eng[df_eng['발행국'].isin(['USPTO', 'US'])])
    data_ep = pd.DataFrame(df_eng[df_eng['발행국'].isin(['EPO', 'EP'])])
    data_cn = pd.DataFrame(df_eng[df_eng['발행국'].isin(['CNIPA', 'CN'])])
    # 출원년구간(10年), 00~09年, 10~20年
    title_us = data_us['title_lemma'].tolist()
    title_ep = data_ep['title_lemma'].tolist()
    title_cn = data_cn['title_lemma'].tolist()
    us_term_freq, us_doc_freq, us_inverse_df = tf_idf(title_us)
    ep_term_freq, ep_doc_freq, ep_inverse_df = tf_idf(title_ep)
    cn_term_freq, cn_doc_freq, cn_inverse_df = tf_idf(title_cn)
    major_pos_us = get_major_pos(data_us['title_lemma1'].tolist())
    major_pos_ep = get_major_pos(data_ep['title_lemma1'].tolist())
    major_pos_cn = get_major_pos(data_cn['title_lemma1'].tolist())
    tf_idf_df_us = pd.DataFrame({'term_freq': pd.Series(us_term_freq),
                                 'doc_freq': pd.Series(us_doc_freq),
                                 'idf': pd.Series(us_inverse_df),
                                 'major_pos': pd.Series(major_pos_us)})
    tf_idf_df_ep = pd.DataFrame({'term_freq': pd.Series(ep_term_freq),
                                 'doc_freq': pd.Series(ep_doc_freq),
                                 'idf': pd.Series(ep_inverse_df),
                                 'major_pos': pd.Series(major_pos_ep)})
    tf_idf_df_cn = pd.DataFrame({'term_freq': pd.Series(cn_term_freq),
                                 'doc_freq': pd.Series(cn_doc_freq),
                                 'idf': pd.Series(cn_inverse_df),
                                 'major_pos': pd.Series(major_pos_cn)})
    tf_idf_df_us['idf_rank'] = tf_idf_df_us['idf'].rank()
    tf_idf_df_ep['idf_rank'] = tf_idf_df_ep['idf'].rank()
    tf_idf_df_cn['idf_rank'] = tf_idf_df_cn['idf'].rank()

    # tf_idf_df['idf_rank'] = tf_idf_df['idf'].rank()
    tf_idf_df_us = tf_idf_df_us[tf_idf_df_us['idf_rank'] >= idf_rank]
    tf_idf_df_ep = tf_idf_df_ep[tf_idf_df_ep['idf_rank'] >= idf_rank]
    tf_idf_df_cn = tf_idf_df_cn[tf_idf_df_cn['idf_rank'] >= idf_rank]

    pass_tag = list(set(tagger.tagdict.values()).difference(stop_pos_list))

    data = data[data['major_pos'].isin(pass_tag)]
    data_us = tf_idf_df_us[tf_idf_df_us['major_pos'].isin(pass_tag)]
    data_ep = tf_idf_df_ep[tf_idf_df_ep['major_pos'].isin(pass_tag)]
    data_cn = tf_idf_df_cn[tf_idf_df_cn['major_pos'].isin(pass_tag)]
    print_step(4, 'Filter with POS Tag')
    print(data_cn.head(5), '\n', title_cn[:5])
    # print(data.head(5))

    # size_x, size_y = 5, 3
    # plt.figure(figsize=(size_x, size_y))  # 단위 : 인치
    #
    # plt.hist(data['idf'])
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()

    # data.to_csv('cleaned_text.csv')
    # if word[1] not in stop_pos_list
    data = data['term_freq'].to_dict()
    data_us = data_us['term_freq'].to_dict()
    data_ep = data_ep['term_freq'].to_dict()
    data_cn = data_cn['term_freq'].to_dict()
    # print(word_cloud_dict)

    # 워드 클라우드 생성
    size_x, size_y = 10, 8
    img = draw_cloud(data=data)
    img_us = draw_cloud(data=data_us)
    img_ep = draw_cloud(data=data_ep)
    img_cn = draw_cloud(data=data_cn)
    # plt.figure(figsize=(size_x, size_y))  # 단위 : 인치
    # fig = plt.figure(figsize=(size_x, size_y))  # 단위 : 인치

    # axes = []

    figure, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    # figure.set_figheight(13)
    # figure.set_figwidth(20)

    ax.ravel()[0].imshow(img_us)
    ax.ravel()[0].set_title('US')
    ax.ravel()[0].set_axis_off()
    ax.ravel()[1].imshow(img_ep)
    ax.ravel()[1].set_title('EP')
    ax.ravel()[1].set_axis_off()
    ax.ravel()[2].imshow(img_cn)
    ax.ravel()[2].set_title('CN')
    ax.ravel()[2].set_axis_off()
    ax.ravel()[3].set_axis_off()
    figure.tight_layout()
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(2, 2, 1)
    # ax.imshow(img_us)
    # ax.set_xlabel('US')
    # ax.set_xticks([]), ax.set_yticks([])
    # ax = fig.add_subplot(2, 2, 2)
    # ax.imshow(img_ep)
    # ax.set_xlabel('EP')
    # ax.set_xticks([]), ax.set_yticks([])
    # ax = fig.add_subplot(2, 2, 3)
    # ax.imshow(img_cn)
    # ax.set_xlabel('CN')
    # ax.set_xticks([]), ax.set_yticks([])
    # fig.tight_layout()
    # plt.show()


    # axes.append(fig.add_subplot(2, 2, 1))
    # axes[-1].set_title('US')

    # plt.imshow(img)
    # plt.tight_layout(pad=0)
    # plt.axis('off')
    # plt.show(block=False)
    # plt.pause(5)

    # img.to_file(os.path.join('img', 'sample_wordcloud.png'))
    # plt.close()

    # idf 변화에 따른 워드클라우드
    # # idf_rank = 0
    # for idf_rank in range(0, 101, 10):
    #     img = draw_cloud(data=tf_idf_df[tf_idf_df['idf_rank'] >= idf_rank])
    #     # wc.to_file(os.path.join('img', 'word_cloud_idf_rank under'+str(idf_rank)+'.png'))
    #     size_x, size_y = 9, 6
    #     plt.figure(figsize=(size_x, size_y))  # 단위 : 인치
    #     plt.imshow(img)
    #     plt.tight_layout(pad=0)
    #     plt.axis('off')
    #     plt.show(block=False)
    #     plt.pause(3)
    #     plt.close()
    # print_step(4, 'Finished to save Wordcloud')


@Timer("examine_counts()")
def examine_counts():
    df_pos = df_eng['title_pos']
    print(df_pos.head(2))
    wordnpos = df_eng['title_pos'].tolist()
    print(wordnpos[0][:4])
    w_list = []
    p_list = []
    for sen in wordnpos:
        for wnp in sen:
            word = lemmatizer.lemmatize(wnp[0].lower(), get_wordnet_pos(wnp[1]))
            w_list.append(word)
            p_list.append(wnp[1])
    wnp_df = pd.DataFrame({'w': np.array(w_list)
                           , 'p': np.array(p_list)
                           # , 'cnt': pd.Series(1, dtype='int32')
                           })
    wnp_df['cnt'] = 1
    print(wnp_df.head(3))
    wnp_df1 = pd.DataFrame(wnp_df.groupby(['w', 'p']).sum())
    print(wnp_df1.head(3))
    # wnp_df2 = pd.DataFrame(wnp_df1.groupby([pd.Grouper(level='w')]).sum())
    wnp_df2 = pd.DataFrame(wnp_df1.groupby(['w'], as_index=True).sum()).reset_index()
    """
    두 문법의 차이를 이해하자. pd.Grouper를 사용하면 그룹화에 사용한 컬럼이 index가 되며 aggregate되며,
    이 방법은 as_index=True와 동일하다.
    as_index=False로 놓고 하면 기존 index를 가져온다.
    wnp_df2 = pd.DataFrame(wnp_df1.groupby([pd.Grouper(level='w')]).sum())
             cnt
    w           
    and     5878
    for     4388
    system  3932
    # wnp_df2 = pd.DataFrame(wnp_df1.groupby(['w'], as_index=False).sum())
           cnt
    250   5878
    1924  4388
    4656  3932
    wnp_df2 = pd.DataFrame(wnp_df1.groupby(['w'], as_index=True).sum())
             cnt
    w           
    and     5878
    for     4388
    system  3932
    """
    print(wnp_df2.sort_values(by='cnt', ascending=False).head(3))
    # print(wnp_df[wnp_df['w'] == 'a'])
    # print(wnp_df1[wnp_df1['w'] == 'a'])
    print(wnp_df2[wnp_df2['w'] == 'a'])
    # print(wnp_df2['a'])


# examine_counts()

"""
단어가 어떻게 사용되는지에 따라 다양한 품사로 출현할텐데, 제거하려는 품사는 tf 카운트에서 제외
             w    p  cnt
0  interactive  NNP    1
1           id  NNP    1
2       system   NN    1
3          use  VBG    1
4       mobile   JJ    1
                 w    p  cnt
0                &   CC    5
1               ''   ''    5
2               's  POS   16
3                (    (  513
4                )    )  513
...            ...  ...  ...
8416  —transparent   NN    1
8417             “   JJ   23
8418             “  NNP    2
8419             ”   NN   18
8420             ”  NNP    7
[8421 rows x 3 columns]

                          w  p  cnt
1052                control  9    9
2974                monitor  8    8
251                  and/or  8    8
3183                   node  8    8
4682                 target  7    7
...                     ... ..  ...
2232  iconology/markerology  1    1
2233                     id  1    1
2234                    ide  1    1
2240    identifier-equipped  1    1
2635                  li-fi  1    1
[5270 rows x 3 columns]

            w    p  cnt
1703  control   JJ    1
1704  control   NN  308
1705  control  NNP  248
1706  control  NNS   11
1707  control   VB   23
1708  control  VBD    3
1709  control  VBG  102
1710  control  VBN    3
1711  control  VBP    1

    w    p   cnt
36  a   DT  1652
37  a   IN    47
38  a  NNP   549
39  a   VB     1

a가 nnp로 발생하는건 어떤 경우일까? pos-tag 오류일까?
a가 vb로 발생한건 1번, proper noun(NNP, 고유명사)로 549번인데, 이러면 idf 순위가 엄청 높아질듯
같이 제거하는게 바람직할까?

    w  p  cnt
35  a  4    4
"""

# print(df_eng[df_eng['title_pos'].isin(['a', 'NNP'])])

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

