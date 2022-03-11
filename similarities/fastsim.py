# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
from typing import List, Union, Tuple, Dict
import os
from loguru import logger
from similarities.similarity import Similarity


class AnnoySimilarity(Similarity):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar query for a given docs with Annoy.
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[str, str]] = None,
            model_name_or_path="shibing624/text2vec-base-chinese",
            embedding_size: int = 768,
            n_trees: int = 256
    ):
        super().__init__(corpus, model_name_or_path)
        self.index = None
        self.embedding_size = embedding_size
        self.n_trees = n_trees
        if corpus is not None and self.corpus_embeddings:
            self.build_index()

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.sentence_model}"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def build_index(self):
        """Build Annoy index after add new documents."""
        # Create Annoy Index
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise ImportError("Annoy is not installed. Please install it first, e.g. with `pip install annoy`.")

        # Creating the annoy index
        self.index = AnnoyIndex(self.embedding_size, 'angular')

        logger.info(f"Init Annoy index, embedding_size: {self.embedding_size}")
        logger.debug(f"Building index with {self.n_trees} trees.")

        for i in range(len(self.corpus_embeddings)):
            self.index.add_item(i, self.corpus_embeddings[i])
        self.index.build(self.n_trees)

    def save_index(self, index_path: str):
        """Save the annoy index to disk."""
        if self.index and index_path:
            logger.info(f"Saving index to: {index_path}")
            self.index.save(index_path)
        else:
            logger.warning("No index path given. Index not saved.")

    def load_index(self, index_path: str):
        """Load Annoy Index from disc."""
        if index_path and os.path.exists(index_path):
            logger.info(f"Loading index from: {index_path}")
            self.index.load(index_path)
        else:
            logger.warning("No index path given. Index not loaded.")

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        result = {}
        if self.corpus_embeddings and self.index is None:
            logger.warning(f"No index found. Please add corpus and build index first, e.g. with `build_index()`."
                           f"Now returning slow search result.")
            return super().most_similar(queries, topn)
        if not self.corpus_embeddings:
            logger.error("No corpus_embeddings found. Please add corpus first, e.g. with `add_corpus()`.")
            return result
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_texts = list(queries.values())
        queries_embeddings = self._get_vector(queries_texts)
        # Annoy get_nns_by_vector can only search for one vector at a time
        for idx, (qid, query) in enumerate(queries.items()):
            corpus_ids, distances = self.index.get_nns_by_vector(queries_embeddings[idx], topn, include_distances=True)
            for corpus_id, distance in zip(corpus_ids, distances):
                score = 1 - (distance ** 2) / 2
                result[qid][self.corpus_ids_map[corpus_id]] = score

        return result


class HnswlibSimilarity(Similarity):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar query for a given docs with Hnswlib.
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[str, str]] = None,
            model_name_or_path="shibing624/text2vec-base-chinese",
            embedding_size: int = 768, ef_construction: int = 400, M: int = 64, ef: int = 50
    ):
        super().__init__(corpus, model_name_or_path)
        self.embedding_size = embedding_size
        self.ef_construction = ef_construction
        self.M = M
        self.ef = ef
        self.index = None
        if corpus is not None and self.corpus_embeddings:
            self.build_index()

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.sentence_model}"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def build_index(self):
        """Build Hnswlib index after add new documents."""
        # Create hnswlib Index
        try:
            import hnswlib
        except ImportError:
            raise ImportError("Hnswlib is not installed. Please install it first, e.g. with `pip install hnswlib`.")

        # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length,
        # then is Inner Product equal to cosine similarity
        self.index = hnswlib.Index(space='cosine', dim=self.embedding_size)
        # Init the HNSWLIB index
        logger.info(f"Creating HNSWLIB index, max_elements: {len(self.corpus)}")
        logger.debug(f"Parameters Required: M: {self.M}")
        logger.debug(f"Parameters Required: ef_construction: {self.ef_construction}")
        logger.debug(f"Parameters Required: ef(>topn): {self.ef}")

        self.index.init_index(max_elements=len(self.corpus_embeddings), ef_construction=self.ef_construction, M=self.M)
        # Then we train the index to find a suitable clustering
        self.index.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))
        # Controlling the recall by setting ef:
        self.index.set_ef(self.ef)  # ef should always be > top_k_hits

    def save_index(self, index_path: str):
        """Save the annoy index to disk."""
        if self.index and index_path:
            logger.info(f"Saving index to: {index_path}")
            self.index.save_index(index_path)
        else:
            logger.warning("No index path given. Index not saved.")

    def load_index(self, index_path: str):
        """Load Annoy Index from disc."""
        if index_path and os.path.exists(index_path):
            logger.info(f"Loading index from: {index_path}")
            self.index.load_index(index_path)
        else:
            logger.warning("No index path given. Index not loaded.")

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        result = {}
        if self.corpus_embeddings and self.index is None:
            logger.warning(f"No index found. Please add corpus and build index first, e.g. with `build_index()`."
                           f"Now returning slow search result.")
            return super().most_similar(queries, topn)
        if not self.corpus_embeddings:
            logger.error("No corpus_embeddings found. Please add corpus first, e.g. with `add_corpus()`.")
            return result
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_texts = list(queries.values())
        queries_embeddings = self._get_vector(queries_texts)
        # We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = self.index.knn_query(queries_embeddings, k=topn)
        # We extract corpus ids and scores for each query
        for i, (qid, query) in enumerate(queries.items()):
            hits = [{'corpus_id': id, 'score': 1 - distance} for id, distance in zip(corpus_ids[i], distances[i])]
            hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            for hit in hits:
                result[qid][self.corpus_ids_map[hit['corpus_id']]] = hit['score']

        return result
