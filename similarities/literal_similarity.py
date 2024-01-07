# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com), Vit Novotny <witiko@mail.muni.cz>, lhy<lhy_in_blcu@126.com>
@description:
Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

This module provides classes that deal with sentence similarities from mean term vector.
Adjust the gensim similarities Index to compute sentence similarities.
"""

import json
import os
from typing import List, Union, Dict

import jieba
import jieba.analyse
import jieba.posseg
import numpy as np
from loguru import logger
from tqdm import tqdm

from similarities.similarity import SimilarityABC
from similarities.utils.distance import string_hash, hamming_distance, longest_match_size
from similarities.utils.rank_bm25 import BM25Okapi
from similarities.utils.tfidf import TFIDF, load_stopwords, default_stopwords_file
from similarities.utils.util import cos_sim, semantic_search

pwd_path = os.path.abspath(os.path.dirname(__file__))


class SimHashSimilarity(SimilarityABC):
    """
    Compute SimHash similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None):
        self.corpus = {}

        self.corpus_embeddings = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: SimHash"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str or dict of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_texts = list(corpus_new.values())
        corpus_embeddings = []
        for sentence in tqdm(corpus_texts, desc="Computing corpus SimHash"):
            corpus_embeddings.append(self.simhash(sentence))
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    def simhash(self, sentence: str):
        """
        Compute SimHash for a given text.
        :param sentence: str
        :return: hash code
        """
        seg = jieba.cut(sentence)
        key_word = jieba.analyse.extract_tags('|'.join(seg), topK=None, withWeight=True, allowPOS=())
        # 先按照权重排序，再按照词排序
        key_list = []
        for feature, weight in key_word:
            weight = int(weight * 20)
            temp = []
            for f in string_hash(feature):
                if f == '1':
                    temp.append(weight)
                else:
                    temp.append(-weight)
            key_list.append(temp)
        content_list = np.sum(np.array(key_list), axis=0)
        # 编码读不出来
        if len(key_list) == 0:
            return '00'
        hash_code = ''
        for c in content_list:
            if c > 0:
                hash_code = hash_code + '1'
            else:
                hash_code = hash_code + '0'
        return hash_code

    def _sim_score(self, seq1, seq2):
        """Convert hamming distance to similarity score."""
        # 将距离转化为相似度
        score = 0.0
        if len(seq1) > 2 and len(seq2) > 2:
            score = 1 - hamming_distance(seq1, seq2, normalize=True)
        return score

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute hamming similarity between two sentences.

        Parameters
        ----------
        a : str or list of str
        b : str or list of str

        Returns
        -------
        list of float
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")
        seqs1 = [self.simhash(text) for text in a]
        seqs2 = [self.simhash(text) for text in b]
        scores = [self._sim_score(seq1, seq2) for seq1, seq2 in zip(seqs1, seqs2)]
        return scores

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute hamming distance between two sentences.

        Parameters
        ----------
        a : str or list of str
        b : str or list of str

        Returns
        -------
        list of float
        """
        sim_scores = self.similarity(a, b)
        return [1 - score for score in sim_scores]

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: list of str or str
        :param topn: int
        :return: dict of dicts, {query_id: {corpus_id, score}, ...}
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            query_emb = self.simhash(query)
            for (corpus_id, doc), doc_emb in zip(self.corpus.items(), self.corpus_embeddings):
                score = self._sim_score(query_emb, doc_emb)
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score

        return result

    def save_corpus_embeddings(self, emb_path: str = "hash_corpus_emb.jsonl"):
        """
        Save corpus embeddings to jsonl file.
        :param emb_path: jsonl file path
        :return:
        """
        with open(emb_path, "w", encoding="utf-8") as f:
            for id, emb in zip(self.corpus.keys(), self.corpus_embeddings):
                json_obj = {"id": id, "doc": self.corpus[id], "doc_emb": list(emb)}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        logger.debug(f"Save corpus embeddings to file: {emb_path}.")

    def load_corpus_embeddings(self, emb_path: str = "hash_corpus_emb.jsonl"):
        """
        Load corpus embeddings from jsonl file.
        :param emb_path: jsonl file path
        :return:
        """
        try:
            with open(emb_path, "r", encoding="utf-8") as f:
                corpus_embeddings = []
                for line in f:
                    json_obj = json.loads(line)
                    self.corpus[int(json_obj["id"])] = json_obj["doc"]
                    corpus_embeddings.append(json_obj["doc_emb"])
                self.corpus_embeddings = corpus_embeddings
        except (IOError, json.JSONDecodeError):
            logger.error("Error: Could not load corpus embeddings from file.")


class TfidfSimilarity(SimilarityABC):
    """
    Compute TFIDF similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None):
        super().__init__()
        self.corpus = {}

        self.corpus_embeddings = []
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

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_texts = list(corpus_new.values())
        corpus_embeddings = []
        for sentence in tqdm(corpus_texts, desc="Computing corpus TFIDF"):
            corpus_embeddings.append(self.tfidf.get_tfidf(sentence))
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute cosine similarity score between two sentences.
        :param a:
        :param b:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        features1 = [self.tfidf.get_tfidf(text) for text in a]
        features2 = [self.tfidf.get_tfidf(text) for text in b]
        return cos_sim(np.array(features1), np.array(features2))

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two keys."""
        return 1 - self.similarity(a, b)

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus.
        :param queries: list of str or str
        :param topn: int
        :return dict of dicts, {query_id: {corpus_id, score}, ...}
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_ids_map = {i: id for i, id in enumerate(list(queries.keys()))}
        queries_texts = list(queries.values())

        queries_embeddings = np.array([self.tfidf.get_tfidf(query) for query in queries_texts], dtype=np.float32)
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topn)
        for idx, hits in enumerate(all_hits):
            for hit in hits[0:topn]:
                result[queries_ids_map[idx]][hit['corpus_id']] = hit['score']

        return result

    def save_corpus_embeddings(self, emb_path: str = "tfidf_corpus_emb.jsonl"):
        """
        Save corpus embeddings to jsonl file.
        :param emb_path: jsonl file path
        :return:
        """
        with open(emb_path, "w", encoding="utf-8") as f:
            for id, emb in zip(self.corpus.keys(), self.corpus_embeddings):
                json_obj = {"id": id, "doc": self.corpus[id], "doc_emb": list(emb)}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        logger.debug(f"Save corpus embeddings to file: {emb_path}.")

    def load_corpus_embeddings(self, emb_path: str = "tfidf_corpus_emb.jsonl"):
        """
        Load corpus embeddings from jsonl file.
        :param emb_path: jsonl file path
        :return:
        """
        try:
            with open(emb_path, "r", encoding="utf-8") as f:
                corpus_embeddings = []
                for line in f:
                    json_obj = json.loads(line)
                    self.corpus[int(json_obj["id"])] = json_obj["doc"]
                    corpus_embeddings.append(json_obj["doc_emb"])
                self.corpus_embeddings = corpus_embeddings
        except (IOError, json.JSONDecodeError):
            logger.error("Error: Could not load corpus embeddings from file.")


class BM25Similarity(SimilarityABC):
    """
    Compute BM25OKapi similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None):
        super().__init__()
        self.corpus = {}

        self.bm25 = None
        self.default_stopwords = load_stopwords(default_stopwords_file)
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

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : dict
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_texts = list(self.corpus.values())
        corpus_seg = [jieba.lcut(d) for d in corpus_texts]
        corpus_seg = [[w for w in doc if (w.strip().lower() not in self.default_stopwords) and
                       len(w.strip()) > 0] for doc in corpus_seg]
        self.bm25 = BM25Okapi(corpus_seg)
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}")

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn=10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: input query
        :param topn: int
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """
        if not self.bm25:
            raise ValueError("BM25 model is not initialized. Please add_corpus first, eg. `add_corpus(corpus)`")
        if isinstance(queries, str):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        for qid, query in queries.items():
            tokens = jieba.lcut(query)
            scores = self.bm25.get_scores(tokens)

            q_res = [{'corpus_id': corpus_id, 'score': score} for corpus_id, score in enumerate(scores)]
            q_res = sorted(q_res, key=lambda x: x['score'], reverse=True)[:topn]
            for res in q_res:
                corpus_id = res['corpus_id']
                result[qid][corpus_id] = res['score']

        return result


class WordEmbeddingSimilarity(SimilarityABC):
    """
    Compute Word2Vec similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None, model_name_or_path="w2v-light-tencent-chinese"):
        """
        Init WordEmbeddingSimilarity.
        :param model_name_or_path: Word2Vec model name or path to model file.
        :param corpus: list of str
        """
        try:
            from text2vec import Word2Vec
        except ImportError:
            raise ImportError("Please install text2vec first, `pip install text2vec`")
        if isinstance(model_name_or_path, str):
            self.keyedvectors = Word2Vec(model_name_or_path)
        elif hasattr(model_name_or_path, "encode"):
            self.keyedvectors = model_name_or_path
        else:
            raise ValueError("model_name_or_path must be ~text2vec.Word2Vec or Word2Vec model name")
        self.corpus = {}

        self.corpus_embeddings = []
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

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_texts = list(corpus_new.values())
        corpus_embeddings = self._get_vector(corpus_texts, show_progress_bar=True).tolist()
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    def _get_vector(self, text, show_progress_bar: bool = False) -> np.ndarray:
        return self.keyedvectors.encode(text, show_progress_bar=show_progress_bar)

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine similarity between two texts."""
        v1 = self._get_vector(a)
        v2 = self._get_vector(b)
        return cos_sim(v1, v2)

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(a, b)

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: list of str or str
        :param topn: int
        :return: dict of dicts, {query_id: {corpus_id, score}, ...}
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_ids_map = {i: id for i, id in enumerate(list(queries.keys()))}
        queries_texts = list(queries.values())

        queries_embeddings = np.array([self._get_vector(query) for query in queries_texts], dtype=np.float32)
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topn)
        for idx, hits in enumerate(all_hits):
            for hit in hits[0:topn]:
                result[queries_ids_map[idx]][hit['corpus_id']] = hit['score']

        return result

    def save_corpus_embeddings(self, emb_path: str = "word2vec_corpus_emb.jsonl"):
        """
        Save corpus embeddings to jsonl file.
        :param emb_path: jsonl file path
        :return: None, save corpus embeddings to file
        """
        with open(emb_path, "w", encoding="utf-8") as f:
            for id, emb in zip(self.corpus.keys(), self.corpus_embeddings):
                json_obj = {"id": id, "doc": self.corpus[id], "doc_emb": list(emb)}
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        logger.debug(f"Save corpus embeddings to file: {emb_path}.")

    def load_corpus_embeddings(self, emb_path: str = "word2vec_corpus_emb.jsonl"):
        """
        Load corpus embeddings from jsonl file.
        :param emb_path: jsonl file path
        :return:
        """
        try:
            with open(emb_path, "r", encoding="utf-8") as f:
                corpus_embeddings = []
                for line in f:
                    json_obj = json.loads(line)
                    self.corpus[int(json_obj["id"])] = json_obj["doc"]
                    corpus_embeddings.append(json_obj["doc_emb"])
                self.corpus_embeddings = corpus_embeddings
        except (IOError, json.JSONDecodeError):
            logger.error("Error: Could not load corpus embeddings from file.")


class CilinSimilarity(SimilarityABC):
    """
    Compute Cilin similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """
    default_cilin_path = os.path.join(pwd_path, 'data/cilin.txt')

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None, cilin_path: str = default_cilin_path):
        super().__init__()
        self.cilin_dict = self.load_cilin_dict(cilin_path)  # Cilin(词林) semantic dictionary
        self.corpus = {}

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

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start add new docs: {len(corpus_new)}")
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}")

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

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute Cilin similarity between two texts.
        :param a:
        :param b:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")

        def calc_pair_sim(sentence1, sentence2):
            words1 = [word.word for word in jieba.posseg.cut(sentence1) if word.flag[0] not in ['u', 'x', 'w']]
            words2 = [word.word for word in jieba.posseg.cut(sentence2) if word.flag[0] not in ['u', 'x', 'w']]
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

        return [calc_pair_sim(sentence1, sentence2) for sentence1, sentence2 in zip(a, b)]

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return [1 - s for s in self.similarity(a, b)]

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            for corpus_id, doc in self.corpus.items():
                score = self.similarity(query, doc)[0]
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score

        return result


class HownetSimilarity(SimilarityABC):
    """
    Compute Hownet similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    """
    default_hownet_path = os.path.join(pwd_path, 'data/hownet.txt')

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None, hownet_path: str = default_hownet_path):
        self.hownet_dict = self.load_hownet_dict(hownet_path)  # semantic dictionary
        self.corpus = {}

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

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start add new docs: {len(corpus_new)}")
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}")

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

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Computer Hownet similarity between two texts.
        :param a:
        :param b:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")

        def calc_pair_sim(sentence1, sentence2):
            words1 = [word.word for word in jieba.posseg.cut(sentence1) if word.flag[0] not in ['u', 'x', 'w']]
            words2 = [word.word for word in jieba.posseg.cut(sentence2) if word.flag[0] not in ['u', 'x', 'w']]
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

        return [calc_pair_sim(sentence1, sentence2) for sentence1, sentence2 in zip(a, b)]

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute Hownet distance between two keys."""
        return [1 - s for s in self.similarity(a, b)]

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            for corpus_id, doc in self.corpus.items():
                score = self.similarity(query, doc)[0]
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score

        return result


class SameCharsSimilarity(SimilarityABC):
    """
    Compute text chars similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    不考虑文本字符位置顺序，基于相同字符数占比计算相似度
    """

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None):
        super().__init__()
        self.corpus = {}

        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: SameChars"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start add new docs: {len(corpus_new)}")
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}")

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute Chars similarity between two texts.
        :param a:
        :param b:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")

        def calc_pair_sim(sentence1, sentence2):
            if not sentence1 or not sentence2:
                return 0.0
            same = set(sentence1) & set(sentence2)
            similarity_score = max(len(same) / len(set(sentence1)), len(same) / len(set(sentence2)))
            return similarity_score

        return [calc_pair_sim(sentence1, sentence2) for sentence1, sentence2 in zip(a, b)]

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return [1 - s for s in self.similarity(a, b)]

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            for corpus_id, doc in self.corpus.items():
                score = self.similarity(query, doc)[0]
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score

        return result


class SequenceMatcherSimilarity(SimilarityABC):
    """
    Compute text sequence matcher similarity between two sentences and retrieves most
    similar sentence for a given corpus.
    考虑文本字符位置顺序，基于最长公共子串占比计算相似度
    """

    def __init__(self, corpus: Union[List[str], Dict[int, str]] = None):
        super().__init__()
        self.corpus = {}

        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: SequenceMatcher"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start add new docs: {len(corpus_new)}")
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}")

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]],
                   min_same_len: int = 70, min_same_len_score: float = 0.9):
        """
        Compute Chars similarity between two texts.
        :param a:
        :param b:
        :param min_same_len:
        :param min_same_len_score:
        :return:
        """
        if isinstance(a, str):
            a = [a]
        if isinstance(b, str):
            b = [b]
        if len(a) != len(b):
            raise ValueError("expected two inputs of the same length")

        def calc_pair_sim(sentence1, sentence2):
            if not sentence1 or not sentence2:
                return 0.0
            same_size = longest_match_size(sentence1, sentence2)
            same_score = min_same_len_score if same_size > min_same_len else 0.0
            similarity_score = max(same_size / len(sentence1), same_size / len(sentence2), same_score)
            return similarity_score

        return [calc_pair_sim(sentence1, sentence2) for sentence1, sentence2 in zip(a, b)]

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return [1 - s for s in self.similarity(a, b)]

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            for corpus_id, doc in self.corpus.items():
                score = self.similarity(query, doc)[0]
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score

        return result
