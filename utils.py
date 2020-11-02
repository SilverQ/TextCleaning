# coding=utf8
import psycopg2 as pg2
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import matplotlib.pyplot as plt
from psycopg2.extras import DictCursor
import time
import numpy as np
import csv

# ImportError: Missing optional dependency 'xlrd'.
# Install xlrd >= 1.0.0 for Excel support Use pip or conda to install xlrd.


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
