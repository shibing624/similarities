# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
from typing import List
from loguru import logger
from similarities.similarity import Similarity


class AnnoySimilarity(Similarity):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar query for a given docs with Annoy.
    """

    def __init__(self, model_name_or_path="shibing624/text2vec-base-chinese", corpus: List[str] = None,
                 embedding_size: int = 384, n_trees: int = 256):
        super().__init__(model_name_or_path, corpus)
        self.index = None
        if corpus is not None and self.corpus_embeddings:
            self.build_index(embedding_size, n_trees)

    def build_index(self, embedding_size: int = 384, n_trees: int = 256):
        """Build Annoy index after add new documents."""
        # Create Annoy Index
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise ImportError("Annoy is not installed. Please install it first, e.g. with `pip install annoy`.")

        # Creating the annoy index
        self.index = AnnoyIndex(embedding_size, 'angular')

        logger.info(f"Init annoy index, embedding_size: {embedding_size}")
        logger.info(f"Building index with {n_trees} trees.")

        for i in range(len(self.corpus_embeddings)):
            self.index.add_item(i, self.corpus_embeddings[i])
        self.index.build(n_trees)

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

    def most_similar(self, query: str, topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        result = []

        query_embeddings = self._get_vector(query)
        if self.corpus_embeddings and self.index is None:
            logger.warning(f"No index found. Please add corpus and build index first, e.g. with `build_index()`."
                           f"Now returning slow search result.")
            return super().most_similar(query, topn)
        if not self.corpus_embeddings:
            logger.error("No corpus_embeddings found. Please add corpus first, e.g. with `add_corpus()`.")
            return result

        corpus_ids, scores = self.index.get_nns_by_vector(query_embeddings, topn, include_distances=True)
        for id, score in zip(corpus_ids, scores):
            score = 1 - ((score ** 2) / 2)
            result.append((id, self.corpus[id], score))

        return result


class HnswlibSimilarity(Similarity):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar query for a given docs with Hnswlib.
    """

    def __init__(self, model_name_or_path="shibing624/text2vec-base-chinese", corpus: List[str] = None,
                 embedding_size: int = 384, ef_construction: int = 400, M: int = 64, ef: int = 50):
        super().__init__(model_name_or_path, corpus)
        self.index = None
        if corpus is not None and self.corpus_embeddings:
            self.build_index(embedding_size, ef_construction, M, ef)

    def build_index(self, embedding_size: int = 384, ef_construction: int = 400, M: int = 64, ef: int = 50):
        """Build Hnswlib index after add new documents."""
        # Create hnswlib Index
        try:
            import hnswlib
        except ImportError:
            raise ImportError("Hnswlib is not installed. Please install it first, e.g. with `pip install hnswlib`.")

        # We use Inner Product (dot-product) as Index. We will normalize our vectors to unit length,
        # then is Inner Product equal to cosine similarity
        self.index = hnswlib.Index(space='cosine', dim=embedding_size)
        # Init the HNSWLIB index
        logger.info(f"Start creating HNSWLIB index, max_elements: {len(self.corpus)}")
        logger.info(f"Parameters Required: M: {M}")
        logger.info(f"Parameters Required: ef_construction: {ef_construction}")
        logger.info(f"Parameters Required: ef(>topn): {ef}")

        self.index.init_index(max_elements=len(self.corpus_embeddings), ef_construction=ef_construction, M=M)
        # Then we train the index to find a suitable clustering
        self.index.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))
        # Controlling the recall by setting ef:
        self.index.set_ef(ef)  # ef should always be > top_k_hits

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

    def most_similar(self, query: str, topn: int = 10):
        """Find the topn most similar texts to the query against the corpus."""
        result = []

        query_embeddings = self._get_vector(query)
        if self.corpus_embeddings and self.index is None:
            logger.warning(f"No index found. Please add corpus and build index first, e.g. with `build_index()`."
                           f"Now returning slow search result.")
            return super().most_similar(query, topn)
        if not self.corpus_embeddings:
            logger.error("No corpus_embeddings found. Please add corpus first, e.g. with `add_corpus()`.")
            return result

        # We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = self.index.knn_query(query_embeddings, k=topn)
        # We extract corpus ids and scores for the first query
        hits = [{'corpus_id': id, 'score': 1 - distance} for id, distance in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
        for hit in hits:
            result.append((hit['corpus_id'], self.corpus[hit['corpus_id']], hit['score']))

        return result
