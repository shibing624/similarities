# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Compute similarity:
1. Compute the similarity between two sentences
2. Retrieves most similar sentence of a query against a corpus of documents.
"""

from typing import List, Union, Tuple

import os
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from similarities.utils.util import cos_sim, semantic_search, dot_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


class SimilarityABC:
    """
    Interface for similarity compute and search.

    In all instances, there is a corpus against which we want to perform the similarity search.
    For each similarity search, the input is a document or a corpus, and the output are the similarities
    to individual corpus documents.
    """

    # def __init__(self, corpus: List[str] = None):
    #     """
    #
    #     Parameters
    #     ----------
    #     corpus : list of str
    #         Corpus of documents.
    #     """
    #     raise NotImplementedError("cannot instantiate Abstract Base Class")

    def add_corpus(self, corpus: List[str]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def similarity(self, text1: Union[str, List[str]], text2: Union[str, List[str]]):
        """
        Compute similarity between two texts.
        :param text1: list of str or str
        :param text2: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def distance(self, text1: Union[str, List[str]], text2: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def most_similar(self, queries: Union[str, List[str]], topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: str
        :param topn: int
        :return: list of list of tuples
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")


class Similarity(SimilarityABC):
    """
    Bert similarity:
    1. Compute the similarity between two sentences
    2. Retrieves most similar sentence of a query against a corpus of documents.

    The index supports adding new documents dynamically.
    """

    def __init__(self, model_name_or_path="shibing624/text2vec-base-chinese", corpus: List[str] = None):
        """
        Initialize the similarity object.
        :param model_name_or_path: The name of the model or the path to the matching model.
        :param corpus: Corpus of documents to use for similarity queries.
        """
        if isinstance(model_name_or_path, str):
            self.sentence_model = SentenceTransformer(model_name_or_path)
        elif hasattr(model_name_or_path, "encode"):
            self.sentence_model = model_name_or_path
        else:
            raise ValueError("model_name_or_path is model name of SentenceTransformer or transformers")
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.corpus = []
        self.corpus_embeddings = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.sentence_model}"
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
        docs_embeddings = self._get_vector(corpus).tolist()
        if self.corpus_embeddings:
            self.corpus_embeddings += docs_embeddings
        else:
            self.corpus_embeddings = docs_embeddings
        logger.info(f"Add docs size: {len(corpus)}, total size: {len(self.corpus)}")

    def _get_vector(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Returns the embeddings for a batch of sentences.
        :param text:
        :return:
        """
        return self.sentence_model.encode(text)

    def similarity(self, text1: Union[str, List[str]], text2: Union[str, List[str]], score_function: str = "cos_sim"):
        """
        Compute similarity between two texts.
        :param text1: list of str or str
        :param text2: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        text_emb1 = self._get_vector(text1)
        text_emb2 = self._get_vector(text2)

        return score_function(text_emb1, text_emb2)

    def distance(self, text1: Union[str, List[str]], text2: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, queries: Union[str, List[str]], topn: int = 10) -> List[List[Tuple[int, str, float]]]:
        """
        Find the topn most similar texts to the queries against the corpus.
        :param queries: str or list of str
        :param topn: int
        :return: list of each query result tuples (corpus_id, corpus_text, similarity_score)
        """
        result = []
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        queries_embeddings = self._get_vector(queries)
        all_hits = semantic_search(queries_embeddings, np.array(self.corpus_embeddings, dtype=np.float32), top_k=topn)
        # logger.debug(f"batch_hits: {batch_hits}")
        for hits in all_hits:
            q_res = []
            for hit in hits[0:topn]:
                q_res.append((hit['corpus_id'], self.corpus[hit['corpus_id']], hit['score']))
            result.append(q_res)

        return result
