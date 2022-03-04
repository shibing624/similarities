# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import jieba
import logging


class JiebaTokenizer(object):
    def __init__(self, dict_path='', custom_word_freq_dict=None):
        self.model = jieba
        self.model.default_logger.setLevel(logging.ERROR)
        # 初始化大词典
        if os.path.exists(dict_path):
            self.model.set_dictionary(dict_path)
        # 加载用户自定义词典
        if custom_word_freq_dict:
            for w, f in custom_word_freq_dict.items():
                self.model.add_word(w, freq=f)

    def tokenize(self, sentence, cut_all=False, HMM=True):
        """
        切词并返回切词位置
        :param sentence: 句子
        :param cut_all: 全模式，默认关闭
        :param HMM: 是否打开NER识别，默认打开
        :return:  A list of strings.
        """
        return self.model.lcut(sentence, cut_all=cut_all, HMM=HMM)
