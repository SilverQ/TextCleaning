# coding=utf8
import os
import time
from utils import *
from konlpy.tag import Okt
from nltk.tokenize import TreebankWordTokenizer, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
# import nltk
# import numpy as np
# import pandas as pd
# from nltk import regexp_tokenize    # 영어 토크나이저
# from collections import Counter, defaultdict
# from math import log  # IDF 계산을 위해
# import pickle
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import spacy

starting_time = time.time()

data_dir = 'metadata'
img_dir = 'img'

# konlpy 설치
# https://ellun.tistory.com/46
# Penn Treebank Tokenization 규칙 - 하이픈으로 구성된 단어는 하나로 유지, doesn't와 같이 어퍼스트로피로 접어가 함께하는 단어는 분
t = Okt()


def proc_time():
    return str(round(time.time() - starting_time, 1)) + ' sec : '


df_excel = pd.read_excel('sna_v2.xlsx', sheet_name='rawdata')
# df = pd.read_excel('sna_v2_10.xlsx', sheet_name='rawdata', encoding='utf-8')
df_excel = df_excel.set_index('Publication Number-발행번호')
print(proc_time(), 'Success to load Excel file with', len(df_excel), 'patents')  # Success to load Excel file with 17094 patents
set_pandas_display_options()

# print(df.columns)     # 컬럼 구성 확인
# print(df.head(1))

# pandas에서 한국 데이터, 미국 데이터를 따로 필터링해서 편집하는 방법은 아직 찾기 어려우므로,
# 데이터를 분리시킨 후 가공을 완료하여 union 하는 것으로 하자!

# df_kipo = df_excel[df_excel['발행국'] == 'KIPO']
# print('Success to identify KR', len(df_kipo), 'patents')       # Success to identify KR 1409 patents
df_kor = df_excel[df_excel['발행국'].isin(['KIPO', 'JPO'])]
print(proc_time(), 'Success to identify KR, JP', len(df_kor), 'patents')    # Success to identify KR, JP 5721 patents
# tokenizer = TreebankWordTokenizer()

# df_eng = df_excel[df_excel['발행국'].isin(['USPTO', 'EPO', 'CNIPA'])]
# 이 상태에서 컬럼을 추가하거나 데이터를 가공하면 SettingWithCopyWarning이 발생하므로, 새로운 df로 만들어주는 것이 바람직하다
df_eng = pd.DataFrame(df_excel[df_excel['발행국'].isin(['USPTO', 'EPO', 'CNIPA'])])
# df_uspto = df_excel[df_excel['발행국'] == 'USPTO']
print(proc_time(), 'Success to identify US, EP, CN', len(df_eng), 'patents')     # Success to identify US, EP, CN 11373 patents
# df_eng는 view와 동일.


def stop_pos():
    # 사용 예정 형태소
    #   FW: foreign word(외국어)
    #   NN: noun, common, singular or mass(명사, 일반, 단수, 질량)
    #   NNP: noun, proper, singular(명사, 고유, 단수)
    #   NNPS: noun, proper, plural(명사, 고유, 복수)
    #   NNS: noun, common, plural(명사, 공통, 복수)
    # 검토 예정 형태소
    #   JJ: adjective or numeral, ordinal(형용사, 숫자, 서수)
    #   JJR: adjective, comparative(형용사, 비교)
    #   JJS: adjective, superlative(형용사, 최상급)
    '''
    사용 예정
        FW: foreign word(외국어)
        NN: noun, common, singular or mass(명사, 일반, 단수, 질량)
        NNP: noun, proper, singular(명사, 고유, 단수)
        NNPS: noun, proper, plural(명사, 고유, 복수)
        NNS: noun, common, plural(명사, 공통, 복수)

    검토 예정
        JJ: adjective or numeral, ordinal(형용사, 숫자, 서수)
        JJR: adjective, comparative(형용사, 비교)
        JJS: adjective, superlative(형용사, 최상급)

    제거 예정
        CC, DT, IN, TO, ,, $, '', (, ), --, ., :
        'CD', 'DT', 'EX', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'

    전체 목록
        FW: foreign word(외국어)
        NN: noun, common, singular or mass(명사, 일반, 단수, 질량)
        NNP: noun, proper, singular(명사, 고유, 단수)
        NNPS: noun, proper, plural(명사, 고유, 복수)
        NNS: noun, common, plural(명사, 공통, 복수)
        RP: particle(입자)
        SYM: symbol(기호)
        None


    전체 목록
        $: dollar
            $ -$ --$ A$ C$ HK$ M$ NZ$ S$ U.S.$ US$
        '': closing quotation mark
            ' ''
        (: opening parenthesis
            ( [ {
        ): closing parenthesis
            ) ] }
        ,: comma
            ,
        --: dash
            --
        .: sentence terminator
            . ! ?
        :: colon or ellipsis
            : ; ...

        CC: conjunction, coordinating(접속사)
            & 'n and both but either et for less minus neither nor or plus so therefore times v. versus vs. whether yet
        CD: numeral, cardinal(숫자)
            mid-1890 nine-thirty forty-two one-tenth ten million 0.5 one forty-seven 1987 twenty '79 zero two 78-degrees eighty-four IX '60s .025
            fifteen 271,124 dozen quintillion DM2,000 ...
        DT: determiner(결정자)
            all an another any both del each either every half la many much nary
            neither no some such that the them these this those
        EX: existential there
            there
        FW: foreign word(외국어)
            gemeinschaft hund ich jeux habeas Haementeria Herr K'ang-si vous
            lutihaw alai je jour objets salutaris fille quibusdam pas trop Monte
            terram fiche oui corporis ...
        IN: preposition or conjunction, subordinating(전치사, 접속사)
            astride among uppon whether out inside pro despite on by throughout
            below within for towards near behind atop around if like until below
            next into if beside ...
        JJ: adjective or numeral, ordinal(형용사, 숫자, 서수)
            third ill-mannered pre-war regrettable oiled calamitous first separable
            ectoplasmic battery-powered participatory fourth still-to-be-named
            multilingual multi-disciplinary ...
        JJR: adjective, comparative(형용사, 비교)
            bleaker braver breezier briefer brighter brisker broader bumper busier
            calmer cheaper choosier cleaner clearer closer colder commoner costlier
            cozier creamier crunchier cuter ...
        JJS: adjective, superlative(형용사, 최상급)
            calmest cheapest choicest classiest cleanest clearest closest commonest
            corniest costliest crassest creepiest crudest cutest darkest deadliest
            dearest deepest densest dinkiest ...
        LS: list item marker(목록 항목 마커)
            A A. B B. C C. D E F First G H I J K One SP-44001 SP-44002 SP-44005
            SP-44007 Second Third Three Two * a b c d first five four one six three
            two
        MD: modal auxiliary(모달 보조)
            can cannot could couldn't dare may might must need ought shall should
            shouldn't will would
        NN: noun, common, singular or mass(명사, 일반, 단수, 질량)
            common-carrier cabbage knuckle-duster Casino afghan shed thermostat
            investment slide humour falloff slick wind hyena override subhumanity
            machinist ...
        NNP: noun, proper, singular(명사, 고유, 단수)
            Motown Venneboerger Czestochwa Ranzer Conchita Trumplane Christos
            Oceanside Escobar Kreisler Sawyer Cougar Yvette Ervin ODI Darryl CTCA
            Shannon A.K.C. Meltex Liverpool ...
        NNPS: noun, proper, plural(명사, 고유, 복수)
            Americans Americas Amharas Amityvilles Amusements Anarcho-Syndicalists
            Andalusians Andes Andruses Angels Animals Anthony Antilles Antiques
            Apache Apaches Apocrypha ...
        NNS: noun, common, plural(명사, 공통, 복수)
            undergraduates scotches bric-a-brac products bodyguards facets coasts
            divestitures storehouses designs clubs fragrances averages
            subjectivists apprehensions muses factory-jobs ...
        PDT: pre-determiner(사전 결정자)
            all both half many quite such sure this
        POS: genitive marker(유전적 마커)
            ' 's
        PRP: pronoun, personal(대명사, 개인)
            hers herself him himself hisself it itself me myself one oneself ours
            ourselves ownself self she thee theirs them themselves they thou thy us
        PRP$: pronoun, possessive(대명사, 소유격)
            her his mine my our ours their thy your
        RB: adverb(부사)
            occasionally unabatingly maddeningly adventurously professedly
            stirringly prominently technologically magisterially predominately
            swiftly fiscally pitilessly ...
        RBR: adverb, comparative(부사, 비교)
            further gloomier grander graver greater grimmer harder harsher
            healthier heavier higher however larger later leaner lengthier less-
            perfectly lesser lonelier longer louder lower more ...
        RBS: adverb, superlative(부사, 최상급)
            best biggest bluntest earliest farthest first furthest hardest
            heartiest highest largest least less most nearest second tightest worst
        RP: particle(입자)
            aboard about across along apart around aside at away back before behind
            by crop down ever fast for forth from go high i.e. in into just later
            low more off on open out over per pie raising start teeth that through
            under unto up up-pp upon whole with you
        SYM: symbol(기호)
            % & ' '' ''. ) ). * + ,. < = > @ A[fj] U.S U.S.S.R * ** ***
        TO: "to" as preposition or infinitive marker(전치사 또는 부정사 마커)
            to
        UH: interjection(감탄사)
            Goodbye Goody Gosh Wow Jeepers Jee-sus Hubba Hey Kee-reist Oops amen
            huh howdy uh dammit whammo shucks heck anyways whodunnit honey golly
            man baby diddle hush sonuvabitch ...
        VB: verb, base form(동사, 기본형)
            ask assemble assess assign assume atone attention avoid bake balkanize
            bank begin behold believe bend benefit bevel beware bless boil bomb
            boost brace break bring broil brush build ...
        VBD: verb, past tense(동사, 과거형)
            dipped pleaded swiped regummed soaked tidied convened halted registered
            cushioned exacted snubbed strode aimed adopted belied figgered
            speculated wore appreciated contemplated ...
        VBG: verb, present participle or gerund(동사, 현재 분사 또는 동명사)
            telegraphing stirring focusing angering judging stalling lactating
            hankerin' alleging veering capping approaching traveling besieging
            encrypting interrupting erasing wincing ...
        VBN: verb, past participle(동사, 과거 분사)
            multihulled dilapidated aerosolized chaired languished panelized used
            experimented flourished imitated reunifed factored condensed sheared
            unsettled primed dubbed desired ...
        VBP: verb, present tense, not 3rd person singular
            predominate wrap resort sue twist spill cure lengthen brush terminate
            appear tend stray glisten obtain comprise detest tease attract
            emphasize mold postpone sever return wag ...
        VBZ: verb, present tense, 3rd person singular
            bases reconstructs marks mixes displeases seals carps weaves snatches
            slumps stretches authorizes smolders pictures emerges stockpiles
            seduces fizzes uses bolsters slaps speaks pleads ...
        WDT: WH-determiner
            that what whatever which whichever
        WP: WH-pronoun
            that what whatever whatsoever which who whom whosoever
        WP$: WH-pronoun, possessive
            whose
        WRB: Wh-adverb
            how however whence whenever where whereby whereever wherein whereof why
        ``: opening quotation mark
            ` ``
        None
    '''
    stop_pos1 = ['CC', 'DT', 'IN', 'TO', ',', '$', ""''"", '(', ')', '--', '.', ':']
    stop_pos2 = ['CC', 'DT', 'IN', 'TO', ',', '$', ""''"", '(', ')', '--', '.', ':', 'CD', 'EX', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'UH']
    stop_pos3 = ['CC', 'DT', 'IN', 'TO', ',', '$', ""''"", '(', ')', '--', '.', ':', 'CD', 'EX', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    return stop_pos1, stop_pos2, stop_pos3


stop_word_set = set(stopwords.words('english'))


def clean_text(input_df, fname, source='발명의명칭', target='title_clean'):
    path = os.path.join(data_dir, fname)
    stop_pos1, stop_pos2, stop_pos3 = stop_pos()
    # input_df['title_clean'] = input_df['발명의명칭'].str.replace('[^a-zA-Z]', ' ')
    # input_df['title_clean'] = input_df['title_clean'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 1]))
    # input_df['title_clean'] = input_df['title_clean'].apply(lambda x: x.lower())
    input_df['title_token'] = input_df['발명의명칭'].apply(lambda x: word_tokenize(x))
    print(proc_time(), 'Success to tokenize')
    input_df['title_pos'] = input_df['title_token'].apply(lambda x: pos_tag(x))
    input_df['title_pos1'] = input_df['title_pos'].apply(lambda x: [word[0] for word in x if word[1] not in stop_pos1])
    input_df['title_pos2'] = input_df['title_pos'].apply(lambda x: [word[0] for word in x if word[1] not in stop_pos2])
    input_df['title_pos3'] = input_df['title_pos'].apply(lambda x: [word[0] for word in x if word[1] not in stop_pos3])
    input_df['title_pos1'] = input_df['title_pos1'].apply(lambda x: [w for w in x if len(w) > 1])
    input_df['title_pos2'] = input_df['title_pos2'].apply(lambda x: [w for w in x if len(w) > 1])
    input_df['title_pos3'] = input_df['title_pos3'].apply(lambda x: [w for w in x if len(w) > 1])
    print(proc_time(), 'Success to filter with pos info')
    input_df.to_csv(path, encoding='utf-8', sep='\t')
    print(proc_time(), 'Success to save file to csv')
    return input_df


df_eng = clean_text(df_eng, 'eng_title_pos_info.csv')

# print("1'st cleaning completed.\n", df_eng[['발명의명칭', 'title_token', 'title_pos']].head(2))
# print("1'st cleaning completed.\n", df_eng[['title_pos', 'title_pos2']].head(20))

title_orig = df_eng['발명의명칭'].tolist()
title_clean1 = df_eng['title_pos1'].tolist()
title_clean2 = df_eng['title_pos2'].tolist()
title_clean3 = df_eng['title_pos3'].tolist()
title_clean1 = [' '.join(sen) for sen in title_clean1]
title_clean2 = [' '.join(sen) for sen in title_clean2]
title_clean3 = [' '.join(sen) for sen in title_clean3]
# print(title_clean)
print(proc_time(), 'Success to extract cleaned title')


def draw_wordcloud(text, f_name):
    wc1 = WordCloud(max_font_size=200, stopwords=stop_word_set, background_color='white',
                    max_words=100,
                    random_state=42,
                    width=800, height=400)
    wc1.generate(' '.join(text))
    plt.figure(figsize=(8, 4))
    plt.imshow(wc1)
    plt.tight_layout(pad=0)
    plt.axis('off')
    # plt.show()
    plt.savefig(os.path.join(img_dir, f_name+'.png'))


# draw_wordcloud(title_clean1, 'title_clean1')
# draw_wordcloud(title_clean2, 'title_clean2')
# draw_wordcloud(title_clean3, 'title_clean3')
# draw_wordcloud(title_orig, 'title_orig')

df_sna = pd.DataFrame(df_eng[['출원년', '출원인/특허권자', 'title_pos3']])
# print(df_sna.head(3))
# of가 눈에 띄는 이유는 형태소 분석 결과에 오류가 있기 때문. IDF를 사용하여 걸러줄 수단을 추가해주자.
# df_sna['title_pos3'] = df_sna['title_pos3'].apply(lambda x: pd.Series(x))
df_sna2 = pd.DataFrame(df_sna.explode('title_pos3'))
print(df_sna2.head(5))
print(proc_time(), 'Success to exploding keyword')
# 정제가 끝난 '발명의 명칭'은 한 셀에 리스트 형태로 들어가 있으며, 이를 열로 내리기 위해 explode 함수를 사용한다.

# print(df_sna2.head(10))

# sql에서 inner join하여 edge를 구하자
df_sna3 = pd.DataFrame(df_sna2.groupby(['출원년', '출원인/특허권자', 'title_pos3']).count())
# df_sna2 = df_sna2.set_index(['출원년', '출원인/특허권자', 'title_pos3'])
print(df_sna3.head(5))
print(proc_time(), 'Success to reduce duplicate')

# df_sna4 = pd.DataFrame(pd.merge(df_sna3, df_sna3, left_on='title_pos3', right_index=True))
# print(df_sna4.head(5))
# print(proc_time(), 'Success to merge table')
#
# df_sna5 = pd.DataFrame(df_sna4.groupby(['출원년_x', '출원년_y', '출원인/특허권자_x', '출원인/특허권자_y']
#                                        , as_index=True).count()).reset_index()
# print(df_sna5.head(5))
# print(proc_time(), 'Success to create aggregate')
# # df_sna4 = df_sna4.set_index(['출원년_x', '출원년_y', '출원인/특허권자_x', '출원인/특허권자_y'])
# df_sna6 = df_sna5[df_sna5['출원년_x'] == df_sna5['출원년_y']]
# df_sna6 = df_sna6[df_sna6['출원인/특허권자_x'] != df_sna6['출원인/특허권자_y']]
# print(df_sna6.head(5))
# print(proc_time(), 'Success to create edge list')


# print(nltk.help.upenn_tagset())

# # SettingWithCopyWarning 발생
# # 필터링한 데이터는 DB의 View와 동일하므로, 여기에 데이터 편집을 수행하는 것은 권장하지 않음.
# # 데이터를 편집하고자 하면, 더 이상 View가 아닌 독립된 데이터 공간을 할당함으로써, 원시 데이터와 완전 분리하는 것을 권함
# # https://emilkwak.github.io/pandas-dataframe-settingwithcopywarning
# df_excel = clean_text(df_excel)
# print("1'st cleaning completed.\n", df_excel[['발명의명칭', 'title_clean']].head(200))


# def cal_similarity(input_df, compare_column='발명의명칭', num_compo=100):
#     title_np = np.array(input_df[compare_column])
#     # print(df.describe())     # 3334건
#     vec = CountVectorizer(min_df=1, encoding='utf-8', stop_words='english')
#     title_dtm = vec.fit_transform(title_np)     # return document-term matrix
#     print('Success to create Term-Doc Matrix')
#     title_tdm = pd.DataFrame(title_dtm.toarray(), index=input_df.index,
#                              columns=vec.get_feature_names(), dtype='d')
#     # print(title_tdm.head(5))    # [3334 rows x 2232 columns] -> stopwords 지정 후 [5 rows x 2155 columns]
#
#     # num_compo = 100
#     # vec.get_feature_names() : Term-doc Matrix 컬럼명이 될 단어들
#     svd_model = TruncatedSVD(n_components=num_compo, algorithm='randomized',
#                              n_iter=100, random_state=122)   # randomized algorithm
#     dtm_lsa = svd_model.fit_transform(title_tdm)
#     # print(dtm_lsa)
#     dtm_lsa = pd.DataFrame(Normalizer(copy=False).fit_transform(dtm_lsa), index=input_df.index)
#     print(dtm_lsa.head())   # 주성분 분해가 완료된 문서 출력
#
#     # # 주성분만 출력해보기
#     # dtm_compo = pd.DataFrame(lsa.components_, index=[i for i in range(num_compo)],
#     #                          columns=vec.get_feature_names())
#     # print(dtm_compo)
#     #
#     # xs = [dtm_lsa.loc[i, 0] for i in dtm_lsa.index]
#     # ys = [dtm_lsa.loc[i, 1] for i in dtm_lsa.index]
#     #
#     # plt.figure()
#     # plt.scatter(xs, ys)
#     # plt.show()
#
#     # Calculate Similarity
#     similarity = np.asarray(np.asmatrix(dtm_lsa) * np.asmatrix(dtm_lsa).T)
#     sim_pd = pd.DataFrame(similarity, index=input_df.index, columns=input_df.index)
#     print(sim_pd.head(5))
#
#     # sim_idx_list = np.array(sim_pd.get('US007976648B1'))
#     # # print(sim_idx_list)
#     # # sim_idx_list.argsort()[::-1]
#     # print(sim_idx_list.argsort()[::-1][1:5])    # [2553 1787  398 2096]
#     # # print(sim_pd.columns(sim_idx_list.argsort()[::-1][1]))
#     # print(sim_pd.columns[0])
#     # print(sim_pd.columns[613])
#     # print(sim_pd.columns[614])
#
#     result = []
#     for i in sim_pd.index:
#         # print(sim_pd[i].name)
#         sim_idx_list = np.array(sim_pd.get(sim_pd[i].name))     # 유사도 목록
#         temp = [sim_pd.columns[i] for i in sim_idx_list.argsort()[::-1][0:10]]
#         # 유사도 내림차순 정렬 후 5개 선택
#         temp.insert(0, sim_pd[i].name)
#         result.append(temp)
#         # print(sim_pd[i])
#     return result
#
#
# sim_list = pd.DataFrame(cal_similarity(df_uspto, '발명의명칭', 100))
# sim_list.to_csv(os.path.join(data_dir, 'sim_pat_list.csv'),
#                 encoding='utf-8', sep='\t')
# print('Success to save')

# print(np.array(sim_list[:5]))
#
# for i in np.array(sim_list[:5]):
#     print(np.array(sim_list)[i, 0])
#     print(df_us['발명의명칭'][np.array(sim_list)[0, 0]])
#     print(np.array(sim_list)[0, 0])
