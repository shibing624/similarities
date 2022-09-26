# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import jieba
import jieba.posseg

from jieba.analyse.tfidf import DEFAULT_IDF, _get_abs_path

pwd_path = os.path.abspath(os.path.dirname(__file__))
default_stopwords_file = os.path.join(pwd_path, '../data/stopwords.txt')


def load_stopwords(file_path):
    stopwords = set()
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                stopwords.add(line)
    return stopwords


class IDFLoader:
    def __init__(self, idf_path=None):
        self.path = ""
        self.idf_freq = {}
        self.median_idf = 0.0
        if idf_path:
            self.set_new_path(idf_path)

    def set_new_path(self, new_idf_path):
        if self.path != new_idf_path:
            self.path = new_idf_path
            content = open(new_idf_path, 'rb').read().decode('utf-8')
            self.idf_freq = {}
            for line in content.splitlines():
                word, freq = line.strip().split(' ')
                self.idf_freq[word] = float(freq)
            self.median_idf = sorted(
                self.idf_freq.values())[len(self.idf_freq) // 2]

    def get_idf(self):
        return self.idf_freq, self.median_idf


class TFIDF:
    def __init__(self, idf_path=None, stopwords=None):
        self.stopwords = stopwords if stopwords is not None else load_stopwords(default_stopwords_file)
        self.idf_loader = IDFLoader(idf_path or DEFAULT_IDF)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def set_idf_path(self, idf_path):
        new_abs_path = _get_abs_path(idf_path)
        if not os.path.isfile(new_abs_path):
            raise Exception("IDF file does not exist: " + new_abs_path)
        self.idf_loader.set_new_path(new_abs_path)
        self.idf_freq, self.median_idf = self.idf_loader.get_idf()

    def get_tfidf(self, sentence):
        words = [word.word for word in jieba.posseg.cut(sentence) if word.flag[0] not in ['u', 'x', 'w']]
        words = [word for word in words if word.lower() not in self.stopwords or len(word.strip()) < 2]
        word_idf = {word: self.idf_freq.get(word, self.median_idf) for word in words}

        res = []
        for w in list(self.idf_freq.keys()):
            res.append(word_idf.get(w, 0))
        return res
