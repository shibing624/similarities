#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Vit Novotny <witiko@mail.muni.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
This module provides classes that deal with term similarities.
Adjust the Index to compute term similarities.
"""
import math
from loguru import logger
from typing import Dict, List, Tuple, Set, Optional, Union
import numpy as np
import torch
import jieba
import jieba.posseg
from text2vec import Word2Vec
from similarities.similarity import cos_sim, Similarity
import os
from similarities.utils.distance import cosine_distance
from simhash import Simhash
from similarities.utils.tfidf import TFIDF

pwd_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WordEmbeddingSimilarity(object):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar terms for a given term.

    Notes
    -----
    By fitting the word embeddings to a vocabulary that you will be using, you
    can eliminate all out-of-vocabulary (OOV) words that you would otherwise
    receive from the `most_similar` method. In subword models such as fastText,
    this procedure will also infer word-vectors for words from your vocabulary
    that previously had no word-vector.

    Parameters
    ----------
    keyedvectors : :class:`~text2vec.Word2Vec`
        The word embeddings.
    docs: list of str
    """

    def __init__(self, keyedvectors: Word2Vec, docs: List[str] = None):
        # super().__init__()
        self.keyedvectors = keyedvectors
        self.docs = []
        self.docs_embeddings = np.array([])
        if docs is not None:
            self.add_documents(docs)

    def __len__(self):
        """Get length of index."""
        return self.docs_embeddings.shape[0]

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def add_documents(self, docs):
        """Extend the index with new documents.

        Parameters
        ----------
        docs : iterable of list of str
        """
        self.docs += docs
        docs_embeddings = self.get_vector(docs)
        if self.docs_embeddings.size > 0:
            self.docs_embeddings = np.vstack((self.docs_embeddings, docs_embeddings))
        else:
            self.docs_embeddings = docs_embeddings
        logger.info(f"Add docs size: {len(docs)}, total size: {len(self.docs)}")

    def get_vector(self, text):
        return self.keyedvectors.encode(text)

    def similarity(self, text1, text2, score_function=cos_sim):
        text_emb1 = self.get_vector(text1)
        text_emb2 = self.get_vector(text2)
        return score_function(text_emb1, text_emb2)

    def distance(self, text1, text2):
        """Compute cosine distance between two keys.
        Calculate 1 - :meth:`~gensim.models.keyedvectors.KeyedVectors.similarity`.

        Parameters
        ----------
        w1 : str
            Input key.
        w2 : str
            Input key.

        Returns
        -------
        float
            Distance between `w1` and `w2`.

        """
        return 1 - self.similarity(text1, text2)

    def semantic_search(
            self,
            query_embeddings: Union[torch.Tensor, np.ndarray],
            corpus_embeddings: Union[torch.Tensor, np.ndarray],
            query_chunk_size: int = 100,
            corpus_chunk_size: int = 500000,
            top_k: int = 10,
            score_function=cos_sim
    ):
        """
        This function performs a cosine similarity search between a list of query embeddings and a list of corpus embeddings.
        It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

        :param query_embeddings: A 2 dimensional tensor with the query embeddings.
        :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
        :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
        :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
        :param top_k: Retrieve top k matching entries.
        :param score_function: Funtion for computing scores. By default, cosine similarity.
        :return: Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the keys 'corpus_id' and 'score'
        """

        if isinstance(query_embeddings, (np.ndarray, np.generic)):
            query_embeddings = torch.from_numpy(query_embeddings)
        elif isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)

        if len(query_embeddings.shape) == 1:
            query_embeddings = query_embeddings.unsqueeze(0)

        if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
            corpus_embeddings = torch.from_numpy(corpus_embeddings)
        elif isinstance(corpus_embeddings, list):
            corpus_embeddings = torch.stack(corpus_embeddings)

        # Check that corpus and queries are on the same device
        query_embeddings = query_embeddings.to(device)
        corpus_embeddings = corpus_embeddings.to(device)

        queries_result_list = [[] for _ in range(len(query_embeddings))]

        for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
            # Iterate over chunks of the corpus
            for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
                # Compute cosine similarity
                cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                            corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

                # Get top-k scores
                cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])),
                                                                           dim=1, largest=True, sorted=False)
                cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
                cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

                for query_itr in range(len(cos_scores)):
                    for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr],
                                                    cos_scores_top_k_values[query_itr]):
                        corpus_id = corpus_start_idx + sub_corpus_id
                        query_id = query_start_idx + query_itr
                        queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

        # Sort and strip to top_k results
        for idx in range(len(queries_result_list)):
            queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
            queries_result_list[idx] = queries_result_list[idx][0:top_k]

        return queries_result_list

    def most_similar(self, query, topn=10):
        result = []
        query_embeddings = self.get_vector(query)
        hits = self.semantic_search(query_embeddings, self.docs_embeddings, top_k=topn)
        hits = hits[0]  # Get the hits for the first query

        print("Input question:", query)
        for hit in hits[0:topn]:
            result.append((self.docs[hit['corpus_id']], round(hit['score'], 4)))
            print("\t{:.3f}\t{}".format(hit['score'], self.docs[hit['corpus_id']]))

        print("\n\n========\n")
        return result


class CilinSimilarity(object):
    """
    Computes cilin similarities between word embeddings and retrieves most
    similar terms for a given term.
    """
    default_cilin_path = os.path.join(pwd_path, 'data', 'cilin.txt')

    def __init__(self, cilin_path: str = default_cilin_path, docs: List[str] = None):
        super().__init__()
        self.cilin_dict = self.load_cilin_dict(cilin_path)  # Cilin(词林) semantic dictionary
        self.docs = []
        if docs is not None:
            self.add_documents(docs)

    def __len__(self):
        """Get length of index."""
        return len(self.docs)

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def add_documents(self, docs):
        """Extend the index with new documents.

        Parameters
        ----------
        docs : iterable of list of str
        """
        self.docs += docs
        logger.info(f"Add docs size: {len(docs)}, total size: {len(self.docs)}")

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

    def _compute_word_sim(self, word1, word2):
        """
        比较计算词语之间的相似度，取max最大值
        :param word1:
        :param word2:
        :return:
        """
        sems_word1 = self.cilin_dict.get(word1, [])
        sems_word2 = self.cilin_dict.get(word2, [])
        score_list = [self._compute_sem(sem_word1, sem_word2) for sem_word1 in sems_word1 for sem_word2 in sems_word2]
        if score_list:
            return max(score_list)
        else:
            return 0

    def _compute_sem(self, sem1, sem2):
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

    def similarity(self, text1, text2):
        """
        基于词相似度计算句子相似度
        :param text1:
        :param text2:
        :return:
        """
        words1 = [word.word for word in jieba.posseg.cut(text1) if word.flag[0] not in ['u', 'x', 'w']]
        words2 = [word.word for word in jieba.posseg.cut(text2) if word.flag[0] not in ['u', 'x', 'w']]
        score_words1 = []
        score_words2 = []
        for word1 in words1:
            score = max(self._compute_word_sim(word1, word2) for word2 in words2)
            score_words1.append(score)
        for word2 in words2:
            score = max(self._compute_word_sim(word2, word1) for word1 in words1)
            score_words2.append(score)
        similarity_score = max(sum(score_words1) / len(words1), sum(score_words2) / len(words2))

        return similarity_score

    def distance(self, text1, text2):
        """Compute cosine distance between two keys."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query, topn=10):
        result = []
        for doc in self.docs:
            score = self.similarity(query, doc)
            result.append((doc, round(score, 4)))
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:topn]


class HownetSimilarity(object):
    """
    Computes hownet similarities between word embeddings and retrieves most
    similar terms for a given term.
    """
    default_hownet_path = os.path.join(pwd_path, 'data', 'hownet.txt')

    def __init__(self, cilin_path: str = default_hownet_path, docs: List[str] = None):
        super().__init__()
        self.hownet_dict = self.load_hownet_dict(cilin_path)  # semantic dictionary
        self.docs = []
        if docs is not None:
            self.add_documents(docs)

    def __len__(self):
        """Get length of index."""
        return len(self.docs)

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def add_documents(self, docs):
        """Extend the index with new documents.

        Parameters
        ----------
        docs : iterable of list of str
        """
        self.docs += docs
        logger.info(f"Add docs size: {len(docs)}, total size: {len(self.docs)}")

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

    def _compute_sem(self, sem1, sem2):
        """计算语义相似度"""
        sem_inter = set(sem1).intersection(set(sem2))
        sem_union = set(sem1).union(set(sem2))
        return float(len(sem_inter)) / float(len(sem_union))

    def _compute_word_sim(self, word1, word2):
        """比较两个词语之间的相似度"""
        DEFS_word1 = self.hownet_dict.get(word1, [])
        DEFS_word2 = self.hownet_dict.get(word2, [])
        scores = [self._compute_sem(DEF_word1, DEF_word2) for DEF_word1 in DEFS_word1 for DEF_word2 in DEFS_word2]
        if scores:
            return max(scores)
        else:
            return 0

    def similarity(self, text1, text2):
        """
        基于词相似度计算句子相似度
        :param text1:
        :param text2:
        :return:
        """
        words1 = [word.word for word in jieba.posseg.cut(text1) if word.flag[0] not in ['u', 'x', 'w']]
        words2 = [word.word for word in jieba.posseg.cut(text2) if word.flag[0] not in ['u', 'x', 'w']]
        score_words1 = []
        score_words2 = []
        for word1 in words1:
            score = max(self._compute_word_sim(word1, word2) for word2 in words2)
            score_words1.append(score)
        for word2 in words2:
            score = max(self._compute_word_sim(word2, word1) for word1 in words1)
            score_words2.append(score)
        similarity_score = max(sum(score_words1) / len(words1), sum(score_words2) / len(words2))

        return similarity_score

    def distance(self, text1, text2):
        """Compute cosine distance between two keys."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query, topn=10):
        result = []
        for doc in self.docs:
            score = self.similarity(query, doc)
            result.append((doc, round(score, 4)))
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:topn]


class SimhashSimilarity(object):
    """
    Computes Simhash similarities between word embeddings and retrieves most
    similar terms for a given term.
    """

    def __init__(self, docs: List[str] = None, hashbits=64):
        super().__init__()
        self.docs = []
        self.hashbits = hashbits
        self.docs_embeddings = np.array([])
        if docs is not None:
            self.add_documents(docs)

    def __len__(self):
        """Get length of index."""
        return len(self.docs)

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def add_documents(self, docs):
        """Extend the index with new documents.

        Parameters
        ----------
        docs : iterable of list of str
        """
        self.docs += docs
        docs_embeddings = []
        for doc in docs:
            doc_emb = self._get_code(doc)
            docs_embeddings.append(doc_emb)
            if len(docs_embeddings) % 10000 == 0:
                logger.debug(f"Progress, add docs size: {len(docs_embeddings)}")
        if self.docs_embeddings.size > 0:
            self.docs_embeddings = np.vstack((self.docs_embeddings, docs_embeddings))
        else:
            self.docs_embeddings = docs_embeddings
        logger.info(f"Add docs size: {len(docs)}, total size: {len(self.docs)}")

    def _hamming_distance(self, code_s1, code_s2):
        """利用64位数，计算海明距离"""
        x = (code_s1 ^ code_s2) & ((1 << self.hashbits) - 1)
        ans = 0
        while x:
            ans += 1
            x &= x - 1
        return ans

    def _get_features(self, string):
        """
        对全文进行分词,提取全文特征,使用词性将虚词等无关字符去重
        :param string:
        :return:
        """
        word_list = [word.word for word in jieba.posseg.cut(string) if
                     word.flag[0] not in ['u', 'x', 'w', 'o', 'p', 'c', 'm', 'q']]
        return word_list

    def _get_code(self, string):
        """对全文进行编码"""
        return Simhash(self._get_features(string)).value

    def similarity(self, text1, text2):
        """
        计算句子间的海明距离
        :param text1:
        :param text2:
        :return:
        """
        code_s1 = self._get_code(text1)
        code_s2 = self._get_code(text2)
        similarity_score = (100 - self._hamming_distance(code_s1, code_s2) * 100 / self.hashbits) / 100

        return similarity_score

    def distance(self, text1, text2):
        """Compute cosine distance between two keys."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query, topn=10):
        result = []
        query_emb = self._get_code(query)
        for doc, doc_emb in zip(self.docs, self.docs_embeddings):
            score = (100 - self._hamming_distance(query_emb, doc_emb) * 100 / self.hashbits) / 100
            result.append((doc, round(score, 4)))
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:topn]


class TfidfSimilarity(object):
    """
    Computes Tfidf similarities between word embeddings and retrieves most
    similar texts for a given text.
    """

    def __init__(self, docs: List[str] = None):
        super().__init__()
        self.docs = []
        self.docs_embeddings = np.array([])
        self.tfidf = TFIDF()
        if docs is not None:
            self.add_documents(docs)

    def __len__(self):
        """Get length of index."""
        return len(self.docs)

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def add_documents(self, docs):
        """Extend the index with new documents.

        Parameters
        ----------
        docs : iterable of list of str
        """
        self.docs += docs
        docs_embeddings = np.array(self.tfidf.get_tfidf(docs))
        if self.docs_embeddings.size > 0:
            self.docs_embeddings = np.vstack((self.docs_embeddings, docs_embeddings))
        else:
            self.docs_embeddings = docs_embeddings
        logger.info(f"Add docs size: {len(docs)}, total size: {len(self.docs)}")

    def similarity(self, text1, text2):
        """
        基于tfidf计算句子间的余弦相似度
        :param text1:
        :param text2:
        :return:
        """
        tfidf_features = self.tfidf.get_tfidf([text1, text2])
        return cosine_distance(np.array(tfidf_features[0]), np.array(tfidf_features[1]))

    def distance(self, text1, text2):
        """Compute cosine distance between two keys."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query, topn=10):
        result = []
        query_emb = np.array(self.tfidf.get_tfidf([query]))
        for doc, doc_emb in zip(self.docs, self.docs_embeddings):
            score = cosine_distance(query_emb, doc_emb)
            result.append((doc, round(score, 4)))
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:topn]


if __name__ == '__main__':
    wm = Word2Vec()
    list_of_docs = ["This is a test1", "This is a test2", "This is a test3"]
    list_of_docs2 = ["that is test4", "that is a test5", "that is a test6"]
    m = WordEmbeddingSimilarity(wm, list_of_docs)
    m.add_documents(list_of_docs2)
    v = m.get_vector("This is a test1")
    print(v[:10], v.shape)
    print(m.similarity("This is a test1", "that is a test5"))
    print(m.distance("This is a test1", "that is a test5"))
    print(m.most_similar("This is a test1"))

    text1 = '周杰伦是一个歌手'
    text2 = '刘若英是个演员'
    m = CilinSimilarity()
    print(m.similarity(text1, text2))
    print(m.distance(text1, text2))
    zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
    m.add_documents(zh_list)
    print(m.most_similar('刘若英是演员'))

    m = HownetSimilarity()
    print(m.similarity(text1, text2))
    print(m.distance(text1, text2))
    zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
    m.add_documents(zh_list)
    print(m.most_similar('刘若英是演员'))

    m = SimhashSimilarity()
    print(m.similarity(text1, text2))
    print(m.distance(text1, text2))
    zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
    m.add_documents(zh_list)
    print(m.most_similar('刘若英是演员'))

    m = TfidfSimilarity()
    print(m.similarity(text1, text2))
    print(m.distance(text1, text2))
    zh_list = ['刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌', '我不是演员吗']
    m.add_documents(zh_list)
    print(m.most_similar('刘若英是演员'))
