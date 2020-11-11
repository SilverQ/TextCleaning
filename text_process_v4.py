from utils import *
from text_utils import *
import os
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from wordcloud import WordCloud
from collections import defaultdict
from PIL import Image

# https://www.scss.tcd.ie/~munnellg/projects/visualizing-text.html 참고
# 하다하다 이건 실패


def get_common_surface_form(original_corpus, stemmer):
    counts = defaultdict(lambda: defaultdict(int))
    surface_forms = {}

    for document in original_corpus:
        for token in document:
            stemmed = stemmer.stem(token.lower())
            counts[stemmed][token.lower()] += 1

    for stemmed, originals in counts.items():
        surface_forms[stemmed] = max(originals,
                                     key=lambda i: originals[i])

    return surface_forms


starting_time = time.time()
set_pandas_display_options()

df_excel = pd.read_excel('IOT_1029_1.xlsx')
df_excel = df_excel.set_index('키위번호')
print(proc_time(starting_time), ' 1. Load Excel file with', len(df_excel), 'patents')  # Success to load Excel file with 17094 patents

print(df_excel.columns)     # 컬럼 구성 확인
print(df_excel.head(1))

df_kor = df_excel[df_excel['발행국'].isin(['KIPO', 'JPO', 'KR', 'JP'])]
print(proc_time(starting_time), ' 2. Identify KR, JP', len(df_kor), 'patents')    # Success to identify KR, JP 5721 patents
df_eng = pd.DataFrame(df_excel[df_excel['발행국'].isin(['USPTO', 'EPO', 'CNIPA', 'US', 'EP', 'CN'])])
print(proc_time(starting_time), ' 3. Identify US, EP, CN', len(df_eng), 'patents')     # Success to identify US, EP, CN 11373 patents

# 분석 범위 한정
nat = 'US'
df_eng = df_eng[df_eng['발행국'].isin(['USPTO', 'US'])]

if '발명의명칭' in df_eng.columns:
    title = df_eng['발명의명칭']
    # title_list = title.head().tolist()
    title_list = title.tolist()
    stemmer = PorterStemmer()
    stemmed_corpus = []
    original_corpus = []
    for title in title_list:
        tokens = word_tokenize(title)
        stemmed = [stemmer.stem(token.lower()) for token in tokens]
        stemmed_corpus.append(stemmed)
        original_corpus.append(tokens)
    dictionary = Dictionary(stemmed_corpus)     # expand가 아니라 append로 2dim list를 입력
    # print('dictionary: ', dictionary)
    counts = get_common_surface_form(original_corpus, stemmer)
    # print('counts: ', counts)
    vectors = [dictionary.doc2bow(text) for text in stemmed_corpus]
    # print('vectors: ', vectors)
    tfidf = TfidfModel(vectors)
    # print('tfidf: ', tfidf)
    weights = tfidf[vectors]
    print('weights1: ', weights)
    weights = [(dictionary[pair[0]], pair[1]) for pair in weights]
    print('weights2: ', weights)
    weights1 = {}
    for w in weights:
        weights1[w[0]] = w[1]
    print('weights3: ', weights1)
    mask_im = np.array(Image.open('mask.png'))
    wc = WordCloud(max_font_size=150, stopwords=stopwords.words("english"), mask=mask_im, background_color='white',
                   max_words=int(1000), random_state=42, width=1024, height=768)
    # wc = WordCloud(
    #     background_color="white",
    #     max_words=2000,
    #     width=1024,
    #     height=720,
    #     stopwords=stopwords.words("english")
    # )
    wc.generate_from_frequencies(weights1)
    wc.to_file(os.path.join('img', "word_cloud.png"))

# elif '키위번호' in df_eng.columns:
#     keynum = df_eng['키위번호']
# elif 'Publication Number - 발행번호' in df_eng.columns:
#     keynum = df_eng['Publication Number - 발행번호']
# else:
#     print('발명의 명칭이나 발행번호를 입력해주세요')
#     pass
