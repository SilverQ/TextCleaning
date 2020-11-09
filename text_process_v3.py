# coding=utf8
import os
from utils import *
from konlpy.tag import Okt
from nltk.tokenize import TreebankWordTokenizer, word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import Normalizer
# import time
# import nltk


def params():
    param_dict = {'data_dir': 'metadata',
                  'img_dir': 'img'}
    return param_dict


param = params()
starting_time = time.time()

df_excel = pd.read_excel('sna_v2.xlsx', sheet_name='rawdata')
# df = pd.read_excel('sna_v2_10.xlsx', sheet_name='rawdata', encoding='utf-8')
df_excel = df_excel.set_index('Publication Number-발행번호')
print(proc_time(starting_time), ' 1. Load Excel file with', len(df_excel), 'patents')  # Success to load Excel file with 17094 patents
set_pandas_display_options()

# print(df.columns)     # 컬럼 구성 확인
# print(df.head(1))

df_kor = df_excel[df_excel['발행국'].isin(['KIPO', 'JPO'])]
print(proc_time(starting_time), ' 2. Identify KR, JP', len(df_kor), 'patents')    # Success to identify KR, JP 5721 patents
df_eng = pd.DataFrame(df_excel[df_excel['발행국'].isin(['USPTO', 'EPO', 'CNIPA'])])
print(proc_time(starting_time), ' 3. Identify US, EP, CN', len(df_eng), 'patents')     # Success to identify US, EP, CN 11373 patents

stop_word_set = set(stopwords.words('english'))
print(stop_word_set)
stop_word_set.add('NICKEL'.lower())
stop_word_set.add('METHOD'.lower())
stop_word_set.add('METHODS'.lower())
stop_word_set.add('ALLOY'.lower())
stop_word_set.add('ALLOYS'.lower())


# 제거할 pos tag를 list 형태로 반환
def stop_pos(level=3):
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
    # print(nltk.help.upenn_tagset())
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
    if level == 1:
        stop_pos = ['CC', 'DT', 'IN', 'TO', ',', '$', ""''"", '(', ')', '--', '.', ':']
    elif level == 2:
        stop_pos = ['CC', 'DT', 'IN', 'TO', ',', '$', ""''"", '(', ')', '--', '.', ':', 'CD', 'EX', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'UH']
    else:
        stop_pos = ['CC', 'DT', 'IN', 'TO', ',', '$', ""''"", '(', ')', '--', '.', ':', 'CD', 'EX', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
    return stop_pos


# 특정 형태소가 제거된 발명의 명칭을 list 형태로 반환
def clean_text(input_df, pos_level=3):
    stop_pos_list = stop_pos(pos_level)     # 1~3으로 입력해서 사용하자, 1은 전치사 등 최소 제거, 2는 부사형 제거, 3은 동사형 제거까지
    # input_df['title_token'] = input_df['발명의명칭'].apply(lambda x: x.lower())
    input_df['title_token'] = input_df['발명의명칭'].apply(lambda x: word_tokenize(x))
    print(proc_time(starting_time), ' 4. tokenize the Title')
    input_df['title_pos'] = input_df['title_token'].apply(lambda x: pos_tag(x))
    input_df['title_clean'] = input_df['title_pos'].apply(
        lambda x: [word[0].lower() for word in x if word[1] not in stop_pos_list]
    )
    input_df['title_clean'] = input_df['title_clean'].apply(lambda x: [w for w in x if w not in stop_word_set])
    input_df['title_clean'] = input_df['title_clean'].apply(lambda x: [w for w in x if len(w) > 1])
    print(proc_time(starting_time), ' 5. Filter with pos info')
    return input_df
    # title_clean3 = [' '.join(sen) for sen in input_df['title_clean'].tolist()]
    # return title_clean3


def draw_wordcloud(text, f_name, action='show'):
    wc1 = WordCloud(max_font_size=150, stopwords=stop_word_set, background_color='white',
                    max_words=200, random_state=42, width=1200, height=800)
    wc1.generate(' '.join(text))
    plt.figure(figsize=(9, 6))
    plt.imshow(wc1)
    plt.tight_layout(pad=0)
    plt.axis('off')
    if action == 'show':
        plt.show(block=False)
        plt.pause(2)
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


def make_animation(input_df, filter, start=2000, end=2020, stride=3, window=5):
    ipo_filter = '발행국'
    ipo_val = 'USPTO'
    df_temp = input_df[input_df[ipo_filter].isin([ipo_val])]
    for year in range(start-window+1, end+1, stride):
        df_temp1 = df_temp[df_temp['출원년'].isin([y for y in range(year, year+window+1)])]
        input_text = [' '.join(sen) for sen in df_temp1['title_clean'].tolist()]
        if len(input_text) > 0:
            print(year, '-', year+window, '\n', df_temp1['Application Number-출원번호'].count())
            draw_wordcloud(input_text, '', 'show')
    pass


cleaned_df = clean_text(df_eng, pos_level=3)
# input_text = [' '.join(sen) for sen in cleaned_df['title_clean'].tolist()]
print(proc_time(starting_time), ' 7. Extract cleaned title')
# draw_wordcloud(input_text, 'test_wordcloud')
# draw_wordcloud(title_clean1, 'title_clean1')
# draw_wordcloud(title_clean2, 'title_clean2')
# draw_wordcloud(title_clean3, 'title_clean3')
# draw_wordcloud(title_orig, 'title_orig')
# save_sna_text()

ani_filter = {'발행국': 'USPTO'}

make_animation(df_eng, filter=ani_filter)
