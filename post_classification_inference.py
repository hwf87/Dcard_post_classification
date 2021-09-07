# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: 'Python 3.9.6 64-bit (''tensorflow'': conda)'
#     name: python3
# ---

import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras_bert import get_custom_objects
from utils import mysqlDatabase
from myBertTools import myBertModel, myTokenizer
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()


def single_label_predict(text):
    text = str(text)[:100]
    x1, x2 = MyTokenizer.encode(first=text)
    x1, x2 = np.array([x1]), np.array([x2])
    outcome = model.predict([x1, x2])
    percentage = outcome.max()
    label = label_dic[outcome.argmax()]
    return label, percentage


def multi_label_predict(text, threshold):
    text = str(text)[:100]
    x1, x2 = MyTokenizer.encode(first=text)
    x1, x2 = np.array([x1]), np.array([x2])
    outcome = model.predict([x1, x2])
    multi_label_set = []
    label_idx = np.where(outcome >= threshold)[1]
    for idx in label_idx:
        multi_label_set.append(label_dic[idx])
    return multi_label_set


if __name__ == '__main__':
    # set parameters
    threshold = 0.5
    pretrained_path = '/Users/jackyfu/Desktop/hwf87_git/bert_wwm/'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    model_path = '/Users/jackyfu/Desktop/hwf87_git/Dcard_post_classification/model_output/dcard_post_multi_cls_bert.h5'
    dic_path = '/Users/jackyfu/Desktop/hwf87_git/Dcard_post_classification/model_output/dcard_cate_multi_label_dic.npy'
    
    ## Load Model
    MyBertModel = myBertModel(pretrained_path, config_path, checkpoint_path, vocab_path)
    token_dict = MyBertModel.get_token_dict()
    MyTokenizer = myTokenizer(token_dict)
    model = load_model(model_path, custom_objects=get_custom_objects())
    label_dic = np.load(dic_path, allow_pickle=True).item()

    ## predict sample
    text = '''請益：愛你的人vs你愛的人
    如題，小女子今年30，已婚，和先生結婚兩年了，我從大學時期就一直暗戀我先生，
    他是我的學長，當時他原本還有一個要好的女朋友，但是被我從中作梗破壞了，
    後來發生了一些事，總算讓我帶球嫁給了先生，但是婚後先生
    '''

    # single_label_predict
    label, confidance = single_label_predict(text)
    print('看板預測: ', label, ' 信心水準: ', confidance)

    # multi_label_predict
    multi_label_set = multi_label_predict(text, threshold)
    print('predict multi-label set: ', multi_label_set)


# ## Performance testing
# import yaml
# with open('config.yml', 'r') as stream:
#         myconfig = yaml.load(stream, Loader=yaml.CLoader)
# database_username = myconfig['mysql_database']['database_username']
# database_password = myconfig['mysql_database']['database_password']
# database_ip       = myconfig['mysql_database']['database_ip']
# database_name     = myconfig['mysql_database']['database_name']
# MysqlDatabase = mysqlDatabase(database_username, database_password, database_ip, database_name)

# sql = '''
# SELECT df.name forums_name, dp.*
# FROM Bigdata.dcard_posts dp
# left join Bigdata.dcard_forums df on dp.forumid = df.id
# WHERE 1=1
# '''
# df = MysqlDatabase.select_table(sql)

# df['text'] = df['title'] + ' ' + df['excerpt'] + ' ' + df['topics']
# df_test = df[df.createdAt >= '2021-09-06'][['forums_name', 'id', 'title', 'excerpt', 'topics', 'text']]
# threshold = 0.5
# df_test['predict_forums_name'] = df_test['text'].progress_apply(multi_label_predict, args=(threshold, ))

# import xlsxwriter
# df_test.to_excel('sample_output.xlsx', engine='xlsxwriter', index=False)


