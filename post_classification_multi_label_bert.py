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

# ## Train Model

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from utils import mysqlDatabase
from myBertTools import myBertModel, myTokenizer
from sklearn.model_selection import train_test_split


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class dataPrepare:
    def __init__(self, model_output_path, label_dic_name, sample_N, test_size, random_state):
        self.model_output_path = model_output_path
        self.label_dic_name = label_dic_name
        self.sample_N = sample_N
        self.test_size = test_size
        self.random_state = random_state
    def get_long_data(self, postid, type, title, excerpt, topics, result):
        for label in topics:
            result.append([postid, type, title, excerpt, label])
    def regex(self, text):
        return text.replace("'", '').replace("[", '').replace("]", '').replace(" ", '').split(',')
    def get_multi_label_set(self, df):
        # 看板名稱
        forum_list = df.name.unique().tolist()
        # 整理樣本數大於N的標籤名單
        label_list = np.concatenate(df.topics.apply(self.regex).tolist())
        df_label = pd.DataFrame.from_dict(dict(Counter(label_list)), orient='index', columns=['count']).reset_index()
        df_label.columns = ['label', 'count']
        df_label = df_label[(df_label['count'] >= sample_N) & (df_label['label'] != '')]
        label_list = df_label.label.tolist()
        # 合併看板名單、標籤名單
        label_set = set(np.concatenate([label_list, forum_list]))
        return label_set
    def get_train_test_split(self, df):
        df_t, df_v,= train_test_split(df, test_size = test_size, random_state = random_state, stratify = df.name)
        df_t['type'] = 'train'
        df_v['type'] = 'valid'
        df = pd.concat([df_t, df_v], sort=True)
        df = df.reset_index(drop=True)
        return df
    def save_label_dic(self, df):
        label_list = list(df.columns)
        label_dic = {i : label_list[i] for i in range(0, len(label_list))}
        label_site = model_output_path + label_dic_name
        np.save(label_site, label_dic)
    def get_data_multi_label(self, MysqlDatabase, sql): 
        df = MysqlDatabase.select_table(sql)  
        # split train/test data set
        df = self.get_train_test_split(df)
        # get multi label set
        label_set = self.get_multi_label_set(df)
        # label preparation
        df.topics = df.topics.apply(self.regex)
        df.apply(lambda x: x.topics.append(x['name']), axis=1)
        df = df[['id', 'name', 'title', 'excerpt', 'topics', 'type']].drop_duplicates(subset = 'id')
        df.topics = df.topics.apply(lambda x: list(set(x)))
        df.topics = df.topics.apply(lambda x: set(x).intersection(label_set))
        # one-hot encoding for multi label
        result = []
        df.apply(lambda x: self.get_long_data(x['id'], x.type, x.title, x.excerpt, x.topics, result), axis=1)
        df = pd.DataFrame(result, columns = ['id', 'type', 'title', 'excerpt', 'label'])
        df['text'] = df['title'] + ' ' + df['excerpt']
        df['value'] = 1
        df1 = pd.pivot_table(df, index=['id', 'text', 'type'], columns='label', values='value', fill_value=0)
        # save label dic
        self.save_label_dic(df1)
        # final training data format
        df = df1.reset_index(drop=False)
        Y_df = df1.reset_index(drop=True)
        return df, Y_df


class dataGenerator:
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = MyTokenizer.encode(first=text)
                y = d[1:]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


if __name__ == '__main__':
    # set parameter
    sample_N = 100
    test_size = 0.1
    random_state = 37
    maxlen = 100
    epochs_N = 3
    os.environ['TF_KERAS'] = '1'
    pretrained_path = '/Users/jackyfu/Desktop/hwf87_git/bert_wwm/'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    label_dic_name = 'dcard_cate_multi_label_dic.npy'
    model_name = 'dcard_post_multi_cls_bert.h5'
    model_output_path = '/Users/jackyfu/Desktop/hwf87_git/Dcard_post_classification/model_output/'
    my_forums_list = ['時事', '網路購物', '股票', '美妝', '工作', '考試', '穿搭', '3C', 'Apple', '感情', 
                      '美食', '理財', '居家生活', '臺灣大學', 'YouTuber']
    with open('config.yml', 'r') as stream:
        myconfig = yaml.load(stream, Loader=yaml.CLoader)
    database_username = myconfig['mysql_database']['database_username']
    database_password = myconfig['mysql_database']['database_password']
    database_ip       = myconfig['mysql_database']['database_ip']
    database_name     = myconfig['mysql_database']['database_name']
    MysqlDatabase = mysqlDatabase(database_username, database_password, database_ip, database_name)
    sql = '''
    SELECT df.name, dp.*
    FROM Bigdata.dcard_posts dp
    left join Bigdata.dcard_forums df on dp.forumid = df.id
    WHERE 1=1
    and df.name in :my_forums_list
    '''
    sql = sql.replace(':my_forums_list', str(tuple(my_forums_list)))
    
    ## prepare data format
    DataPrepare = dataPrepare(model_output_path, label_dic_name, sample_N, test_size, random_state)
    df, Y_df = DataPrepare.get_data_multi_label(MysqlDatabase, sql)

    ## build model
    MyBertModel = myBertModel(pretrained_path, config_path, checkpoint_path, vocab_path)
    token_dict = MyBertModel.get_token_dict()
    model = MyBertModel.build_model(Y_df)
    MyTokenizer = myTokenizer(token_dict)

    ## Training and save model
    train_data = df[df['type'] == 'train'].drop(columns=['id', 'type']).values.tolist()
    valid_data = df[df['type'] == 'valid'].drop(columns=['id', 'type']).values.tolist()
    train_D = dataGenerator(train_data)
    valid_D = dataGenerator(valid_data)
    history = model.fit(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=epochs_N,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D)
    )
    model.save(model_output_path + model_name)

    # plot training history
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()




