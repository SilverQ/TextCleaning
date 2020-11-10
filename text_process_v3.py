# coding=utf8
import os
from utils import *
from konlpy.tag import Okt
from nltk.tokenize import TreebankWordTokenizer, word_tokenize
from nltk.tag import pos_tag
# from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from wordcloud import WordCloud
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
import imageio
# import time
# import nltk


def params():
    param_dict = {'data_dir': 'metadata',
                  'img_dir': 'img'}
    return param_dict


param = params()
starting_time = time.time()

# df_excel = pd.read_excel('sna_v2.xlsx', sheet_name='rawdata')
# df_excel = df_excel.set_index('Publication Number-발행번호')
df_excel = pd.read_excel('IOT_1029_1.xlsx')
df_excel = df_excel.set_index('키위번호')
print(proc_time(starting_time), ' 1. Load Excel file with', len(df_excel), 'patents')  # Success to load Excel file with 17094 patents
set_pandas_display_options()

print(df_excel.columns)     # 컬럼 구성 확인
print(df_excel.head(1))

df_kor = df_excel[df_excel['발행국'].isin(['KIPO', 'JPO', 'KR', 'JP'])]
print(proc_time(starting_time), ' 2. Identify KR, JP', len(df_kor), 'patents')    # Success to identify KR, JP 5721 patents
df_eng = pd.DataFrame(df_excel[df_excel['발행국'].isin(['USPTO', 'EPO', 'CNIPA', 'US', 'EP', 'CN'])])
print(proc_time(starting_time), ' 3. Identify US, EP, CN', len(df_eng), 'patents')     # Success to identify US, EP, CN 11373 patents

stop_word_set = set(stopwords.words('english'))
print(stop_word_set)
# stop_word_set.add('NICKEL'.lower())
# stop_word_set.add('METHOD'.lower())
# stop_word_set.add('METHODS'.lower())
# stop_word_set.add('ALLOY'.lower())
# stop_word_set.add('ALLOYS'.lower())
stop_word_set.add('device'.lower())
stop_word_set.add('method'.lower())
stop_word_set.add('system'.lower())
stop_word_set.add('apparatus'.lower())
stop_word_set.add('service'.lower())
stop_word_set.add('network'.lower())
stop_word_set.add('devices'.lower())
stop_word_set.add('thereof'.lower())
stop_word_set.add('internet'.lower())
stop_word_set.add('using'.lower())
stop_word_set.add('methods'.lower())
stop_word_set.add('systems'.lower())
# stop_word_set.add(''.lower())
# stop_word_set.add(''.lower())
# stop_word_set.add(''.lower())
# stop_word_set.add(''.lower())
# stop_word_set.add(''.lower())
# stop_word_set.add(''.lower())


# 특정 형태소가 제거된 발명의 명칭을 list 형태로 반환
def clean_text(input_df, pos_level=3):
    n = WordNetLemmatizer()
    stop_pos_list = stop_pos(pos_level)     # 1~3으로 입력해서 사용하자, 1은 전치사 등 최소 제거, 2는 부사형 제거, 3은 동사형 제거까지
    # input_df['title_token'] = input_df['발명의명칭'].apply(lambda x: x.lower())
    input_df['title_token'] = input_df['발명의명칭'].apply(lambda x: word_tokenize(x))
    print(proc_time(starting_time), ' 4. tokenize the Title')
    input_df['title_pos'] = input_df['title_token'].apply(lambda x: pos_tag(x))
    input_df['title_clean'] = input_df['title_pos'].apply(
        lambda x: [n.lemmatize(word[0], get_wordnet_pos(word[1])).lower() for word in x if word[1] not in stop_pos_list]
    )
    input_df['title_clean'] = input_df['title_clean'].apply(lambda x: [w for w in x if w not in stop_word_set])
    input_df['title_clean'] = input_df['title_clean'].apply(lambda x: [w.upper() for w in x if len(w) > 1])
    print(proc_time(starting_time), ' 5. Filter with pos info')
    print(input_df[['발명의명칭', 'title_pos', 'title_clean']].head(5))
    return input_df
    # title_clean3 = [' '.join(sen) for sen in input_df['title_clean'].tolist()]
    # return title_clean3


def draw_wordcloud(text, f_name, action='show'):
    fig_x, fig_y = 1200, 800
    wc1 = WordCloud(max_font_size=int(fig_y/3), stopwords=stop_word_set, background_color='white',
                    max_words=int(fig_y/4), random_state=42, width=fig_x, height=fig_y)
    wc1.generate(' '.join(text))
    plt.figure(figsize=(9, 6))
    plt.imshow(wc1)
    plt.tight_layout(pad=0)
    plt.axis('off')
    if action == 'show':
        plt.show(block=False)
        plt.pause(1)
        plt.close()
    else:
        plt.savefig(os.path.join(param['img_dir'], f_name+'.png'))


def sna_bipartite(input_df, node_col, timestamp_col, edge_col):
    df_sna = input_df[[timestamp_col, node_col, edge_col]]
    df_sna2 = pd.DataFrame(df_sna.explode(edge_col))
    print(proc_time(starting_time), ' 8. Exploding keyword')
    # Timestamp 데이터 형태 : <[2017.0]>
    # df_sna2['Timestamp'] = df_sna2[timestamp_col].apply(lambda x: '<['+str(round(float(x), 2))+']>')
    df_sna2['Timestamp'] = df_sna2[timestamp_col].apply(lambda x: '['+str(round(float(x), 2))+']')
    print('열 내리기\n', df_sna2.head(2))

    # sql에서 inner join하여 edge를 구하자
    df_sna2['weight'] = 1
    df_sna3 = df_sna2.groupby(['Timestamp', node_col, edge_col]).count().reset_index()
    print('단어 빈도수 산출\n', df_sna3.head(2))
    print(proc_time(starting_time), ' 9. Count of words\n')

    df_sna4 = pd.merge(df_sna3, df_sna3, how='inner', left_on=edge_col, right_on=edge_col)
    # print('self join\n', df_sna4.head(2))
    print(proc_time(starting_time), '10. Merge table\n')

    df_sna5 = df_sna4[df_sna4['Timestamp_x'] == df_sna4['Timestamp_y']]
    df_sna5 = df_sna5[df_sna5[node_col+'_x'] != df_sna5[node_col+'_y']]
    df_sna5['edge'] = (df_sna5['weight_x'] + df_sna5['weight_y']) / 2
    # print(df_sna5.head(5))
    print(proc_time(starting_time), '11. Prepare edge list\n')

    df_sna6 = df_sna5[[edge_col, 'Timestamp_x', node_col+'_x', node_col+'_y', 'edge']]
    df_edge = df_sna6.groupby(['Timestamp_x', node_col+'_x', node_col+'_y']).sum('edge').reset_index()
    df_edge.columns = ['Timestamp', 'Source', 'Target', 'Weight']
    df_edge['Type'] = 'Undirected'
    print(df_sna6.describe(), '\n', df_sna6.head(5))
    print(proc_time(starting_time), '12. Create edge list\n')

    df_node = pd.DataFrame(df_sna2[['Timestamp', node_col]])
    df_node = df_node.reset_index()
    print(df_node.describe(), '\n', df_node.head(2))
    df_node = df_node.groupby(['Timestamp', node_col]).count().reset_index()
    print(df_node.describe(), '\n', df_node.head(4))
    df_node['Id'] = df_node[node_col]
    df_node.columns = ['Timestamp', 'Id', 'Cnt', 'Label']
    print(df_node.describe(), '\n', df_node.head(6))
    return df_node, df_edge


def save_sna_text():
    node, edge = sna_bipartite(df_eng, node_col='출원인/특허권자', timestamp_col='출원년', edge_col='title2')
    node.to_csv(os.path.join(param['data_dir'], 'r_node.csv'), encoding='utf-8', sep='\t', index=False)
    edge.to_csv(os.path.join(param['data_dir'], 'r_edge.csv'), encoding='utf-8', sep='\t', index=False)
    print(proc_time(starting_time), '13. Save edge list\n')


def make_animation(input_df, filter, start=2000, end=2020, stride=1, window=5, padding=0):
    ipo_filter = '발행국'
    ipo_val = ['USPTO', 'US']
    df_temp = input_df[input_df[ipo_filter].isin(ipo_val)]
    for year in range(start-padding, end+1, stride):
        df_temp1 = df_temp[df_temp['출원년'].isin([y for y in range(year, year+window+1)])]
        input_text = [' '.join(sen) for sen in df_temp1['title_clean'].tolist()]
        if len(input_text) > 0:
            print(year, '-', year+window, '\n', df_temp1['출원년'].count())
            draw_wordcloud(input_text, 'USPTO_'+str(year)+'-'+str(year+window), 'show-')
        # pass


cleaned_df = clean_text(df_eng, pos_level=3)
# input_text = [' '.join(sen) for sen in cleaned_df['title_clean'].tolist()]
print(proc_time(starting_time), ' 7. Extract cleaned title')
# draw_wordcloud(input_text, 'test_wordcloud')
# draw_wordcloud(title_clean1, 'title_clean1')
# save_sna_text()

cloud_filter = {'발행국': 'USPTO'}

# make_animation(df_eng, filter=ani_filter, start=2000, end=2020, stride=1, window=5, padding=0)

f_name = 'IOT_1029_1.xlsx'

make_animation(df_eng, filter=cloud_filter,
               start=2010, end=2020, stride=5, window=5, padding=0)
