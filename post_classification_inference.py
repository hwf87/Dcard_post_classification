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
from myBertTools import myBertModel, myTokenizer
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()


def class_predict_fuc(text):
    text = str(text)[:100]
    x1, x2 = MyTokenizer.encode(first=text)
    x1, x2 = np.array([x1]), np.array([x2])
    outcome = model.predict([x1, x2])
    percentage = outcome.max()
    label = label_dic[outcome.argmax()]
    return label, percentage


if __name__ == '__main__':
    pretrained_path = '/Users/jackyfu/Desktop/hwf87_git/bert_wwm/'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    model_path = '/Users/jackyfu/Desktop/hwf87_git/Dcard_post_classification/model_output/dcard_post_cls_bert.h5'
    dic_path = '/Users/jackyfu/Desktop/hwf87_git/Dcard_post_classification/model_output/dcard_cate_label_dic.npy'
    ## Load Model
    MyBertModel = myBertModel(pretrained_path, config_path, checkpoint_path, vocab_path)
    token_dict = MyBertModel.get_token_dict()
    MyTokenizer = myTokenizer(token_dict)
    model = load_model(model_path, custom_objects=get_custom_objects())
    label_dic = np.load(dic_path, allow_pickle=True).item()
    ## predict
    text = '''關於80人以上遠距的信 真是疑問滿滿… 
    1.所以第一週遠距的課要怎麼加簽？
    2.是選課上限80以上的就確定第一週遠距，上限80以下第一週不遠距嗎？還是看選課人數？
    3.那如果看選課人數
    '''
    label, confidance = class_predict_fuc(text)
    print('看板預測: ', label, ' 信心水準: ', confidance)
