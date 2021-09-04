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
import codecs
import tensorflow as tf
import keras_bert
from keras.layers import *
from keras.models import Model
import keras.backend as K
from tensorflow.keras.optimizers import Adam
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras_bert.loader import load_trained_model_from_checkpoint
from keras_bert.bert import *
from keras.utils.vis_utils import plot_model


class myBertModel():
    def __init__(self, pretrained_path, config_path, checkpoint_path, vocab_path):
        self.pretrained_path = pretrained_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.vocab_path = vocab_path
    def get_token_dict(self):
        token_dict = {}
        with codecs.open(self.vocab_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        return token_dict
    def build_model(self, Y_df):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, seq_len=None)

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)
        p = Dense(Y_df.shape[1], activation='sigmoid')(x)

        model = Model([x1_in, x2_in], p)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5), # 用足够小的学习率
            metrics=['accuracy']
        )
        # model.summary()
        return model


class myTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R
