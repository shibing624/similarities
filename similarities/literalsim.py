# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Vit Novotny <witiko@mail.muni.cz>, lhy<lhy_in_blcu@126.com>
@description:
Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

This module provides classes that deal with sentence similarities from mean term vector.
Adjust the gensim similarities Index to compute sentence similarities.
"""

import os
from typing import List, Union

import jieba
import jieba.analyse
import jieba.posseg
import numpy as np
from text2vec import Word2Vec
from loguru import logger
from similarities.utils.distance import cosine_distance
from similarities.utils.distance import sim_hash, hamming_distance
from similarities.utils.rank_bm25 import BM25Okapi
from similarities.utils.tfidf import TFIDF

pwd_path = os.path.abspath(os.path.dirname(__file__))


class SimhashSimilarity:
    """
    Compute SimHash similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: List[str] = None):
        self.corpus = []
        self.corpus_embeddings = np.array([])
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: Simhash"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: List[str]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        self.corpus += corpus
        corpus_embeddings = []
        for sentence in corpus:
            corpus_embeddings.append(self.simhash(sentence))
            if len(corpus_embeddings) % 1000 == 0:
                logger.debug(f"Progress, add corpus size: {len(corpus_embeddings)}")
        if self.corpus_embeddings.size > 0:
            self.corpus_embeddings = np.vstack((self.corpus_embeddings, corpus_embeddings))
        else:
            self.corpus_embeddings = np.array(corpus_embeddings)
        logger.info(f"Add corpus size: {len(corpus)}, total size: {len(self.corpus)}")

    def simhash(self, text: str):
        """
        Compute SimHash for a given text.
        :param text: str
        :return: hash code
        """
        return sim_hash(text)

    def _sim_score(self, v1, v2):
        """Compute hamming similarity between two embeddings."""
        return (100 - hamming_distance(v1, v2) * 100 / 64) / 100

    def similarity(self, text1: str, text2: str):
        """
        Compute hamming similarity between two texts.
        :param text1:
        :param text2:
        :return:
        """
        v1 = self.simhash(text1)
        v2 = self.simhash(text2)
        similarity_score = self._sim_score(v1, v2)

        return similarity_score

    def distance(self, text1: str, text2: str):
        """Compute cosine distance between two keys."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query: str, topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param query: str
        :param topn: int
        :return: list of tuples (text, similarity)
        """
        result = []
        query_emb = self.simhash(query)
        for (corpus_id, doc), doc_emb in zip(enumerate(self.corpus), self.corpus_embeddings):
            score = self._sim_score(query_emb, doc_emb)
            result.append((corpus_id, doc, score))
        result.sort(key=lambda x: x[2], reverse=True)
        return result[:topn]


class TfidfSimilarity:
    """
    Compute TFIDF similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: List[str] = None):
        super().__init__()
        self.corpus = []
        self.corpus_embeddings = np.array([])
        self.tfidf = TFIDF()
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: Tfidf"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: List[str]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        self.corpus += corpus
        corpus_embeddings = []
        for sentence in corpus:
            corpus_embeddings.append(self.tfidf.get_tfidf(sentence))
            if len(corpus_embeddings) % 1000 == 0:
                logger.debug(f"Progress, add corpus size: {len(corpus_embeddings)}")
        if self.corpus_embeddings.size > 0:
            self.corpus_embeddings = np.vstack((self.corpus_embeddings, corpus_embeddings))
        else:
            self.corpus_embeddings = np.array(corpus_embeddings)
        logger.info(f"Add corpus size: {len(corpus)}, total size: {len(self.corpus)}")

    def similarity(self, text1: str, text2: str):
        """
        Compute cosine similarity score between two sentences.
        :param text1:
        :param text2:
        :return:
        """
        feature1, feature2 = self.tfidf.get_tfidf(text1), self.tfidf.get_tfidf(text2)
        return cosine_distance(np.array(feature1), np.array(feature2))

    def distance(self, text1: str, text2: str):
        """Compute cosine distance between two keys."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query: str, topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        result = []
        query_emb = self.tfidf.get_tfidf(query)
        for (corpus_id, doc), doc_emb in zip(enumerate(self.corpus), self.corpus_embeddings):
            score = cosine_distance(query_emb, doc_emb, normalize=True)
            result.append((corpus_id, doc, score))
        result.sort(key=lambda x: x[2], reverse=True)
        return result[:topn]


class BM25Similarity:
    """
    Compute BM25OKapi similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: List[str] = None):
        super().__init__()
        self.corpus = []
        self.bm25 = None
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: BM25"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: List[str]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        self.corpus += corpus
        corpus_seg = [jieba.lcut(d) for d in self.corpus]
        self.bm25 = BM25Okapi(corpus_seg)
        logger.info(f"Add corpus size: {len(corpus)}, total size: {len(self.corpus)}")

    def similarity(self, text1, text2):
        """
        Compute similarity score between two sentences.
        :param text1:
        :param text2:
        :return:
        """
        raise NotImplementedError()

    def distance(self, text1, text2):
        """Compute distance between two sentences."""
        raise NotImplementedError()

    def most_similar(self, query, topn=10):
        tokens = jieba.lcut(query)
        if not self.bm25:
            raise ValueError("BM25 model is not initialized. Please add_corpus first, eg. `add_corpus(corpus)`")
        scores = self.bm25.get_scores(tokens)
        result = [(corpus_id, self.corpus[corpus_id], score) for corpus_id, score in enumerate(scores)]
        result.sort(key=lambda x: x[2], reverse=True)
        return result[:topn]


class WordEmbeddingSimilarity:
    """
    Compute Word2Vec similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, keyedvectors, corpus: List[str] = None):
        """
        Init WordEmbeddingSimilarity.
        :param keyedvectors: ~text2vec.Word2Vec
        :param corpus: list of str
        """
        if isinstance(keyedvectors, Word2Vec):
            self.keyedvectors = keyedvectors
        elif isinstance(keyedvectors, str):
            self.keyedvectors = Word2Vec(keyedvectors)
        else:
            raise ValueError("keyedvectors must be ~text2vec.Word2Vec or Word2Vec model name")
        self.corpus = []
        self.corpus_embeddings = np.array([])
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: Word2Vec"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: List[str]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        self.corpus += corpus
        corpus_embeddings = self.get_vector(corpus)
        if self.corpus_embeddings.size > 0:
            self.corpus_embeddings = np.vstack((self.corpus_embeddings, corpus_embeddings))
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add corpus size: {len(corpus)}, total size: {len(self.corpus)}")

    def get_vector(self, text):
        return self.keyedvectors.encode(text)

    def similarity(self, text1: str, text2: str):
        """Compute cosine similarity between two texts."""
        v1 = self.get_vector(text1)
        v2 = self.get_vector(text2)
        return cosine_distance(v1, v2)

    def distance(self, text1: str, text2: str):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query: str, topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param query: str
        :param topn: int
        :return:
        """
        result = []
        query_emb = self.get_vector(query)
        for (corpus_id, doc), doc_emb in zip(enumerate(self.corpus), self.corpus_embeddings):
            score = cosine_distance(query_emb, doc_emb, normalize=True)
            result.append((corpus_id, doc, score))
        result.sort(key=lambda x: x[2], reverse=True)
        return result[:topn]


class CilinSimilarity:
    """
    Compute Cilin similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """
    default_cilin_path = os.path.join(pwd_path, 'data/cilin.txt')

    def __init__(self, cilin_path: str = default_cilin_path, corpus: List[str] = None):
        super().__init__()
        self.cilin_dict = self.load_cilin_dict(cilin_path)  # Cilin(词林) semantic dictionary
        self.corpus = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: Cilin"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: List[str]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        self.corpus += corpus
        logger.info(f"Add corpus size: {len(corpus)}, total size: {len(self.corpus)}")

    @staticmethod
    def load_cilin_dict(path):
        """加载词林语义词典"""
        sem_dict = {}
        for line in open(path, 'r', encoding='utf-8'):
            line = line.strip()
            terms = line.split(' ')
            sem_type = terms[0]
            words = terms[1:]
            for word in words:
                if word not in sem_dict:
                    sem_dict[word] = sem_type
                else:
                    sem_dict[word] += ';' + sem_type

        for word, sem_type in sem_dict.items():
            sem_dict[word] = sem_type.split(';')
        return sem_dict

    def _word_sim(self, word1, word2):
        """
        比较计算词语之间的相似度，取max最大值
        :param word1:
        :param word2:
        :return:
        """
        sems_word1 = self.cilin_dict.get(word1, [])
        sems_word2 = self.cilin_dict.get(word2, [])
        score_list = [self._semantic_sim(sem_word1, sem_word2) for sem_word1 in sems_word1 for sem_word2 in sems_word2]
        if score_list:
            return max(score_list)
        else:
            return 0

    def _semantic_sim(self, sem1, sem2):
        """
        基于语义计算词语相似度
        :param sem1:
        :param sem2:
        :return:
        """
        sem1 = [sem1[0], sem1[1], sem1[2:4], sem1[4], sem1[5:7], sem1[-1]]
        sem2 = [sem2[0], sem2[1], sem2[2:4], sem2[4], sem2[5:7], sem2[-1]]
        score = 0
        for index in range(len(sem1)):
            if sem1[index] == sem2[index]:
                if index in [0, 1]:
                    score += 3
                elif index == 2:
                    score += 2
                elif index in [3, 4]:
                    score += 1
        return score / 10

    def similarity(self, text1: str, text2: str):
        """
        Compute Cilin similarity between two texts.
        :param text1:
        :param text2:
        :return:
        """
        words1 = [word.word for word in jieba.posseg.cut(text1) if word.flag[0] not in ['u', 'x', 'w']]
        words2 = [word.word for word in jieba.posseg.cut(text2) if word.flag[0] not in ['u', 'x', 'w']]
        score_words1 = []
        score_words2 = []
        for word1 in words1:
            score = max(self._word_sim(word1, word2) for word2 in words2)
            score_words1.append(score)
        for word2 in words2:
            score = max(self._word_sim(word2, word1) for word1 in words1)
            score_words2.append(score)
        similarity_score = max(sum(score_words1) / len(words1), sum(score_words2) / len(words2))

        return similarity_score

    def distance(self, text1: str, text2: str):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query: str, topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        result = []
        for corpus_id, doc in enumerate(self.corpus):
            score = self.similarity(query, doc)
            result.append((corpus_id, doc, score))
        result.sort(key=lambda x: x[2], reverse=True)
        return result[:topn]


class HownetSimilarity:
    """
    Compute Hownet similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """
    default_hownet_path = os.path.join(pwd_path, 'data/hownet.txt')

    def __init__(self, hownet_path: str = default_hownet_path, corpus: List[str] = None):
        self.hownet_dict = self.load_hownet_dict(hownet_path)  # semantic dictionary
        self.corpus = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: Hownet"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: List[str]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        self.corpus += corpus
        logger.info(f"Add corpus size: {len(corpus)}, total size: {len(self.corpus)}")

    @staticmethod
    def load_hownet_dict(path):
        """加载Hownet语义词典"""
        hownet_dict = {}
        for line in open(path, 'r', encoding='utf-8'):
            words = [word for word in line.strip().replace(' ', '>').replace('\t', '>').split('>') if word != '']
            word = words[0]
            word_def = words[2]
            hownet_dict[word] = word_def.split(',')
        return hownet_dict

    def _semantic_sim(self, sem1, sem2):
        """计算语义相似度"""
        sem_inter = set(sem1).intersection(set(sem2))
        sem_union = set(sem1).union(set(sem2))
        return float(len(sem_inter)) / float(len(sem_union))

    def _word_sim(self, word1, word2):
        """比较两个词语之间的相似度"""
        sems_word1 = self.hownet_dict.get(word1, [])
        sems_words = self.hownet_dict.get(word2, [])
        scores = [self._semantic_sim(sem_word1, sem_word2) for sem_word1 in sems_word1 for sem_word2 in sems_words]
        if scores:
            return max(scores)
        else:
            return 0

    def similarity(self, text1: str, text2: str):
        """
        Computer Hownet similarity between two texts.
        :param text1:
        :param text2:
        :return:
        """
        words1 = [word.word for word in jieba.posseg.cut(text1) if word.flag[0] not in ['u', 'x', 'w']]
        words2 = [word.word for word in jieba.posseg.cut(text2) if word.flag[0] not in ['u', 'x', 'w']]
        score_words1 = []
        score_words2 = []
        for word1 in words1:
            score = max(self._word_sim(word1, word2) for word2 in words2)
            score_words1.append(score)
        for word2 in words2:
            score = max(self._word_sim(word2, word1) for word1 in words1)
            score_words2.append(score)
        similarity_score = max(sum(score_words1) / len(words1), sum(score_words2) / len(words2))

        return similarity_score

    def distance(self, text1: str, text2: str):
        """Compute cosine distance between two keys."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query: str, topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        result = []
        for corpus_id, doc in enumerate(self.corpus):
            score = self.similarity(query, doc)
            result.append((corpus_id, doc, score))
        result.sort(key=lambda x: x[2], reverse=True)
        return result[:topn]
