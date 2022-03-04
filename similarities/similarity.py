# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

from typing import List, Union, Optional
import numpy as np
import scipy
from loguru import logger
import torch

import logging
import scipy.sparse
from gensim import utils, matutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from enum import Enum, unique


def cos_sim(v1: Union[torch.Tensor, np.ndarray], v2: Union[torch.Tensor, np.ndarray]):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(v1, torch.Tensor):
        v1 = torch.tensor(v1)
    if not isinstance(v2, torch.Tensor):
        v2 = torch.tensor(v2)
    if len(v1.shape) == 1:
        v1 = v1.unsqueeze(0)
    if len(v2.shape) == 1:
        v2 = v2.unsqueeze(0)

    v1_norm = torch.nn.functional.normalize(v1, p=2, dim=1)
    v2_norm = torch.nn.functional.normalize(v2, p=2, dim=1)
    return torch.mm(v1_norm, v2_norm.transpose(0, 1))


class EncoderType(Enum):
    FIRST_LAST_AVG = 0
    LAST_AVG = 1
    CLS = 2
    POOLER = 3
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError()


class Similarity:
    """
    Compute cosine similarity of a dynamic query against a corpus of documents ('the index').

    The index supports adding new documents dynamically.
    """

    def __init__(self, model_name_or_path=None, docs=None):
        """

        Parameters
        ----------
        output_prefix : str
            Prefix for shard filename. If None, a random filename in temp will be used.
        docs : iterable of list of (int, number)
            Corpus in streamed Gensim bag-of-words format.
        """
        self.model_name_or_path = model_name_or_path
        self.model = None
        logger.debug(f'Loading model {model_name_or_path}')
        logger.debug(f"Device: {device}")

        self.normalize = True
        self.keyedvectors = None
        self.docs = docs
        self.norm = False
        if docs is not None:
            self.add_documents(docs)

    def __len__(self):
        """Get length of index."""
        return self.docs.shape[0]

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def add_documents(self, corpus):
        """Extend the index with new documents.

        Parameters
        ----------
        corpus : iterable of list of (int, number)
            Corpus in BoW format.
        """
        for doc in corpus:
            self.docs.append(doc)
            if len(self.docs) % 10000 == 0:
                logger.info("PROGRESS: fresh_shard size=%i", len(self.docs))

    def get_vector(self, text, norm=False):
        """Get the key's vector, as a 1D numpy array.

        Parameters
        ----------

        text : str
            Key for vector to return.
        norm : bool, optional
            If True, the resulting vector will be L2-normalized (unit Euclidean length).

        Returns
        -------

        numpy.ndarray
            Vector for the specified key.

        Raises
        ------

        KeyError
            If the given key doesn't exist.

        """
        pass

    def similarity(
            self, text1: Union[List[str], str], text2: Union[List[str], str]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Compute similarity between two list of texts.
        :param text1: list, sentence1 list
        :param text2: list, sentence2 list
        :return: return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not text1 or not text2:
            return np.array([])
        if isinstance(text1, str):
            text1 = [text1]  # type: ignore
        if isinstance(text2, str):
            text2 = [text2]  # type: ignore
        pass

    def distance(self, text1: Union[List[str], str], text2: Union[List[str], str]):
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

    def most_similar(self, query: Union[List[str], str], topn=10, threshold=0, exponent=2.0):
        """
        Get topn similar text
        :param query: str, query text
        :param top_k: int, top_k
        :return: list, top_k similar text
        """
        if query not in self.keyedvectors:
            logger.debug('an out-of-dictionary term "%s"', query)
        else:
            most_similar = self.keyedvectors.most_similar(query, topn=topn)
            for t2, similarity in most_similar:
                if similarity > threshold:
                    yield (t2, similarity ** exponent)

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
