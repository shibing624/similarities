# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
Ensemble retriever that ensemble the results of
multiple retrievers by using weighted Reciprocal Rank Fusion
"""

import os
from typing import List, Union, Dict

import torch

from similarities.bert_similarity import BertSimilarity
from similarities.literal_similarity import BM25Similarity
from similarities.similarity import SimilarityABC

pwd_path = os.path.abspath(os.path.dirname(__file__))


class EnsembleSimilarity(SimilarityABC):
    """
    Compute similarity score between two sentences and retrieves most
    similar sentence for a given corpus.
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[int, str]] = None,
            similarities: List[SimilarityABC] = None,
            weights: List[float] = None,
            c: int = 60,
    ):
        """
        Init EnsembleSimilarity.
        :param corpus: A docs list
        :param similarities: A list of Similarity to ensemble
        :param weights: A list of weights corresponding to the similarities. Defaults to equal
            weighting for all similarities.
        :param c: A constant added to the rank, controlling the balance between the importance
            of high-ranked items and the consideration given to lower-ranked items.
            Default is 60.
        """
        self.corpus = {}
        if similarities is None:
            similarities = [BertSimilarity, BM25Similarity]
        if weights is None:
            weights = [0.5, 0.5]
        self.similarities = similarities
        self.weights = weights
        if len(self.similarities) != len(self.weights):
            raise ValueError("The number of similarities and weights must be equal")
        self.c = c
        if corpus is not None:
            self.add_corpus(corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: EnsembleSimilarity"
        base += f"({', '.join([str(i) for i in self.similarities])})"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str or dict of str
        """
        for i in self.similarities:
            i.add_corpus(corpus)
        self.corpus = self.similarities[0].corpus

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute ensemble similarity between two sentences.

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

        similarity_scores = [0] * len(a)

        # Calculate similarity score for each pair and similarity method
        for m, weight in zip(self.similarities, self.weights):
            if hasattr(m, "similarity"):
                # Compute similarities in batch
                batch_similarity_scores = m.similarity(a, b)
                scores = []
                if isinstance(batch_similarity_scores, torch.Tensor):
                    for i in range(len(a)):
                        scores.append(batch_similarity_scores.numpy()[i][i])
                else:
                    scores = batch_similarity_scores
                # Add weighted batch similarity scores to total similarity_scores
                similarity_scores = [s + weight * batch_s for s, batch_s in
                                     zip(similarity_scores, scores)]

        return similarity_scores

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        Compute ensemble cosine distance between two sentences.

        Parameters
        ----------
        a : str or list of str
        b : str or list of str

        Returns
        -------
        list of float
        """
        # For similarity scores, the corresponding distance is 1 - similarity
        return [1 - sim_score for sim_score in self.similarity(a, b)]

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10) -> List[List[Dict]]:
        """
        Find the topn most similar texts to the query against the corpus.
        :param queries: list of str or str
        :param topn: int
        :return: List[List[Dict]], A list with one entry for each query. Each entry is a list of
            dict with the keys 'corpus_id', 'corpus_doc' and 'score', sorted by decreasing cosine similarity scores.
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = []
        # Calculate weighted reciprocal rank fusion for each query
        for qid, query in queries.items():
            # Store RRF scores for each document in corpus
            rrf_scores = {}

            # Calculate RRF scores for each similarity method
            for similarity, weight in zip(self.similarities, self.weights):
                top_docs = similarity.most_similar(query, topn=topn)

                # For each similar document, calculate its RRF score
                if top_docs and len(top_docs) > 0:
                    for doc_scores in top_docs:
                        for rank, v in enumerate(doc_scores):
                            doc_id = v["corpus_id"]
                            score = v["score"]
                            rrf_score = weight / (rank + self.c)
                            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score
            # Order by scores and get only topn
            sorted_by_score = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:topn]
            q_res = []
            for doc_id, score in sorted_by_score:
                q_res.append({
                    "corpus_id": doc_id,
                    "corpus_doc": self.corpus.get(doc_id, ""),
                    "score": score
                })
            result.append(q_res)
        return result

    def save_corpus_embeddings(self, emb_dir: str = "corpus_embs"):
        """
        Save corpus embeddings to jsonl file.
        :param emb_dir: jsonl file dir
        :return:
        """
        os.makedirs(emb_dir, exist_ok=True)
        for i in self.similarities:
            if hasattr(i, "save_corpus_embeddings"):
                save_path = os.path.join(emb_dir, f"{i.__class__.__name__}_corpus_emb.jsonl")
                i.save_corpus_embeddings(save_path)

    def load_corpus_embeddings(self, emb_dir: str = "corpus_embs"):
        """
        Load corpus embeddings from jsonl file.
        :param emb_dir: jsonl file dir
        :return:
        """
        corpus = None
        for i in self.similarities:
            if hasattr(i, "load_corpus_embeddings"):
                load_path = os.path.join(emb_dir, f"{i.__class__.__name__}_corpus_emb.jsonl")
                i.load_corpus_embeddings(load_path)
                corpus = i.corpus
                if not self.corpus:
                    self.corpus = corpus
        for i in self.similarities:
            if not hasattr(i, "load_corpus_embeddings") and corpus:
                i.corpus = corpus
