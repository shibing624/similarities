# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
from typing import List, Union, Dict

from loguru import logger

from similarities.bert_similarity import BertSimilarity


class AnnoySimilarity(BertSimilarity):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar query for a given docs with Annoy.
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[str, str]] = None,
            model_name_or_path="shibing624/text2vec-base-chinese",
            n_trees: int = 256,
            device: str = None
    ):
        super().__init__(corpus, model_name_or_path, device=device)
        self.index = None
        self.embedding_size = self.get_sentence_embedding_dimension()
        self.n_trees = n_trees
        if corpus is not None and self.corpus_embeddings:
            self.build_index()

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.sentence_model}"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def create_index(self):
        """Create Annoy Index."""
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise ImportError("Annoy is not installed. Please install it first, e.g. with `pip install annoy`.")

        # Creating the annoy index
        self.index = AnnoyIndex(self.embedding_size, 'angular')
        logger.debug(f"Init Annoy index, embedding_size: {self.embedding_size}")

    def build_index(self):
        """Build Annoy index after add new documents."""
        self.create_index()
        logger.debug(f"Building index with {self.n_trees} trees.")

        for i in range(len(self.corpus_embeddings)):
            self.index.add_item(i, self.corpus_embeddings[i])
        self.index.build(self.n_trees)

    def save_index(self, index_path: str = "annoy_index.bin"):
        """Save the annoy index to disk."""
        if index_path:
            if self.index is None:
                self.build_index()
            self.index.save(index_path)
            corpus_emb_json_path = index_path + ".jsonl"
            super().save_corpus_embeddings(corpus_emb_json_path)
            logger.info(f"Saving Annoy index to: {index_path}, corpus embedding to: {corpus_emb_json_path}")
        else:
            logger.warning("No index path given. Index not saved.")

    def load_index(self, index_path: str = "annoy_index.bin"):
        """Load Annoy Index from disc."""
        if index_path and os.path.exists(index_path):
            corpus_emb_json_path = index_path + ".jsonl"
            logger.info(f"Loading index from: {index_path}, corpus embedding from: {corpus_emb_json_path}")
            super().load_corpus_embeddings(corpus_emb_json_path)
            if self.index is None:
                self.create_index()
            self.index.load(index_path)
        else:
            logger.warning("No index path given. Index not loaded.")

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10,
                     score_function: str = "cos_sim", **kwargs):
        """Find the topn most similar texts to the query against the corpus."""
        result = {}
        if self.corpus_embeddings and self.index is None:
            logger.warning(f"No index found. Please add corpus and build index first, e.g. with `build_index()`."
                           f"Now returning slow search result.")
            return super().most_similar(queries, topn, score_function=score_function)
        if not self.corpus_embeddings:
            logger.error("No corpus_embeddings found. Please add corpus first, e.g. with `add_corpus()`.")
            return result
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_texts = list(queries.values())
        queries_embeddings = self.get_embeddings(queries_texts, **kwargs)
        # Annoy get_nns_by_vector can only search for one vector at a time
        for idx, (qid, query) in enumerate(queries.items()):
            corpus_ids, distances = self.index.get_nns_by_vector(queries_embeddings[idx], topn, include_distances=True)
            for corpus_id, distance in zip(corpus_ids, distances):
                score = 1 - (distance ** 2) / 2
                result[qid][corpus_id] = score

        return result


class HnswlibSimilarity(BertSimilarity):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar query for a given docs with Hnswlib.
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[str, str]] = None,
            model_name_or_path="shibing624/text2vec-base-chinese",
            ef_construction: int = 400, M: int = 64, ef: int = 50,
            device: str = None,
    ):
        super().__init__(corpus, model_name_or_path, device=device)
        self.embedding_size = self.get_sentence_embedding_dimension()
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

    def create_index(self):
        """Create Hnswlib Index."""
        try:
            import hnswlib
        except ImportError:
            raise ImportError("Hnswlib is not installed. Please install it first, e.g. with `pip install hnswlib`.")

        # Creating the hnswlib index
        self.index = hnswlib.Index(space='cosine', dim=self.embedding_size)
        self.index.init_index(max_elements=len(self.corpus_embeddings), ef_construction=self.ef_construction, M=self.M)
        # Controlling the recall by setting ef:
        self.index.set_ef(self.ef)  # ef should always be > top_k_hits
        logger.debug(f"Init Hnswlib index, embedding_size: {self.embedding_size}")

    def build_index(self):
        """Build Hnswlib index after add new documents."""
        # Init the HNSWLIB index
        self.create_index()
        logger.info(f"Building HNSWLIB index, max_elements: {len(self.corpus)}")
        logger.debug(f"Parameters Required: M: {self.M}")
        logger.debug(f"Parameters Required: ef_construction: {self.ef_construction}")
        logger.debug(f"Parameters Required: ef(>topn): {self.ef}")

        # Then we train the index to find a suitable clustering
        self.index.add_items(self.corpus_embeddings, list(range(len(self.corpus_embeddings))))

    def save_index(self, index_path: str = "hnswlib_index.bin"):
        """Save the index to disk."""
        if index_path:
            if self.index is None:
                self.build_index()
            self.index.save_index(index_path)
            corpus_emb_json_path = index_path + ".jsonl"
            super().save_corpus_embeddings(corpus_emb_json_path)
            logger.info(f"Saving hnswlib index to: {index_path}, corpus embedding to: {corpus_emb_json_path}")
        else:
            logger.error("No index path given. Index not saved.")

    def load_index(self, index_path: str = "hnswlib_index.bin"):
        """Load Index from disc."""
        if index_path and os.path.exists(index_path):
            corpus_emb_json_path = index_path + ".jsonl"
            logger.info(f"Loading index from: {index_path}, corpus embedding from: {corpus_emb_json_path}")
            super().load_corpus_embeddings(corpus_emb_json_path)
            if self.index is None:
                self.create_index()
            self.index.load_index(index_path)
        else:
            logger.warning("No index path given. Index not loaded.")

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10,
                     score_function: str = "cos_sim", **kwargs):
        """Find the topn most similar texts to the query against the corpus."""
        result = {}
        if self.corpus_embeddings and self.index is None:
            logger.warning(f"No index found. Please add corpus and build index first, e.g. with `build_index()`."
                           f"Now returning slow search result.")
            return super().most_similar(queries, topn, score_function=score_function)
        if not self.corpus_embeddings:
            logger.error("No corpus_embeddings found. Please add corpus first, e.g. with `add_corpus()`.")
            return result
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_texts = list(queries.values())
        queries_embeddings = self.get_embeddings(queries_texts, **kwargs)
        # We use hnswlib knn_query method to find the top_k_hits
        corpus_ids, distances = self.index.knn_query(queries_embeddings, k=topn)
        # We extract corpus ids and scores for each query
        for i, (qid, query) in enumerate(queries.items()):
            hits = [{'corpus_id': id, 'score': 1 - distance} for id, distance in zip(corpus_ids[i], distances[i])]
            hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            for hit in hits:
                result[qid][hit['corpus_id']] = hit['score']

        return result
