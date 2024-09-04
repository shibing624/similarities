# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Compute similarity:
1. Compute the similarity between two sentences
2. Retrieves most similar sentence of a query against a corpus of documents.
"""

from typing import List, Union, Dict
from PIL import Image


class SimilarityABC:
    """
    Interface for similarity compute and search.

    In all instances, there is a corpus against which we want to perform the similarity search.
    For each similarity search, the input is a document or a corpus, and the output are the similarities
    to individual corpus documents.
    """

    def add_corpus(self, corpus: Union[List, Dict]):
        """
        Extend the corpus with new documents.
        corpus : list of str
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def similarity(self, a: Union[str, Image.Image, List], b: Union[str, Image.Image, List]):
        """
        Compute similarity between two texts.
        :param a: list of str or str
        :param b: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def distance(self, a: Union[str, Image.Image, List], b: Union[str, Image.Image, List]):
        """Compute cosine distance between two texts."""
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def most_similar(self, queries: Union[str, Image.Image, List, Dict], topn: int = 10) -> List[List[Dict]]:
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: Dict[int(query_id), str(query_text)] or List[str] or str
        :param topn: int
        :return: List[List[Dict]], A list with one entry for each query. Each entry is a list of
            dict with the keys 'corpus_id', 'corpus_doc' and 'score', sorted by decreasing similarity scores.
        """
        raise NotImplementedError("cannot instantiate Abstract Base Class")

    def search(self, queries: Union[str, Image.Image, List, Dict], topn: int = 10) -> List[List[Dict]]:
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: Dict[int(query_id), str(query_text)] or List[str] or str
        :param topn: int
        :return: List[List[Dict]], A list with one entry for each query. Each entry is a list of
            dict with the keys 'corpus_id', 'corpus_doc' and 'score', sorted by decreasing similarity scores.
        """
        return self.most_similar(queries, topn=topn)
