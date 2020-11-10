# coding=utf8
import psycopg2 as pg2
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from psycopg2.extras import DictCursor
import time
import numpy as np
from nltk.corpus import stopwords, wordnet
# import csv

# ImportError: Missing optional dependency 'xlrd'.
# Install xlrd >= 1.0.0 for Excel support Use pip or conda to install xlrd.


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # return None  # for easy if-statement
        return wordnet.NOUN  # for easy if-statement


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


def proc_time(starting_time):
    return str(round(time.time() - starting_time, 1)) + ' sec : '


def set_pandas_display_options() -> None:
    display = pd.options.display
    display.max_columns = 100
    display.max_rows = 100
    display.max_colwidth = 199
    display.width = None


def plot_hist(df):
    ax = df.plot.hist(density=False, alpha=0.9, bins=int((10 / 2)))
    # plt.axis([0, 10, 0, 13000])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.show()
    # plt.savefig(os.path.join(data_dir, 'idf_hist.png'))


def read_xlsx(filename='sna_v2_10.xlsx', sheetname='rawdata'):
    df = pd.read_excel(filename, sheet_name=sheetname)
    df = df[df['발행국'] == 'USPTO']
    for line in df['발명의명칭'].tolist():
        yield line
    pass


def connect(conn_info, application_name=''):
    conn = pg2.connect(conn_info)
    # this is setting for ERROR, invalid byte sequence for encoding "UTF8": 0x00
    # conn.cursor().execute("SET standard_conforming_strings=on")
    if len(application_name) > 0:
        cur = conn.cursor()
        cur.execute("SET application_name TO {0}".format(application_name))
        cur.close()
    return conn


def escape_string(raw):
    return raw.replace("\"", "\\\"").replace("'", "''")


def execute(cur, query):
    cur.execute(query)
    return cur.fetchall()


def conn_string():
    conn_string = "dbname={dbname} user={user} host={host} password={password} port={port}".format(dbname='db_ipc',
                                                                                                   user='scorpion',
                                                                                                   host='pgdb05.nips.local',
                                                                                                   password='scorpion',
                                                                                                   port=5432)
    return conn_string


# if file_type == 'DB':
#     conn = connect(conn_string())
#     cur = conn.cursor()
#     pr_id = proj_id(cur, pr_name)
#     input_type = 'ops'
#     legal_select_query = "select id_kipi, reg_no, reg_date, legal_stat, legal_stat_code from biz.t_stat_doc where project_id = {pr_id}".format(pr_id=pr_id)
#     results = execute(cur, legal_select_query)
# #     print(legal_select_query)
# elif file_type == 'txt' or file_type == 'csv':
#     # quotechar로 감싼 txt 파일에서 작업할 때
# #     pr_name = '200427-PVD법적상태_확인용'
#     with open(pr_name+'.'+file_type, newline='', mode='r', encoding='utf-8') as read_file:
#         reader = csv.reader(read_file, delimiter='\t', quotechar='"')
#         results = list(reader)
