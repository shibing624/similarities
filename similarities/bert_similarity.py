# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Compute similarity:
1. Compute the similarity between two sentences
2. Retrieves most similar sentence of a query against a corpus of documents.
"""

import json
import os
from typing import List, Union, Dict

import numpy as np
from loguru import logger
from text2vec import SentenceModel

from similarities.similarity import SimilarityABC
from similarities.utils.util import cos_sim, semantic_search, dot_score

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"


class BertSimilarity(SimilarityABC):
    """
    Sentence Similarity:
    1. Compute the similarity between two sentences
    2. Retrieves most similar sentence of a query against a corpus of documents.

    The index supports adding new documents dynamically.
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[int, str]] = None,
            model_name_or_path: str = "shibing624/text2vec-base-chinese",
            device: str = None,
    ):
        """
        Initialize the similarity object.
        :param model_name_or_path: Transformer model name or path, like:
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'bert-base-uncased', 'bert-base-chinese',
             'shibing624/text2vec-base-chinese', ...
            model in HuggingFace Model Hub and release from https://github.com/shibing624/text2vec
        :param corpus: Corpus of documents to use for similarity queries.
        :param device: Device (like 'cuda' / 'cpu') to use for the computation.
        """
        if isinstance(model_name_or_path, str):
            self.sentence_model = SentenceModel(
                model_name_or_path,
                device=device
            )
        elif hasattr(model_name_or_path, "encode"):
            self.sentence_model = model_name_or_path
        else:
            raise ValueError("model_name_or_path is transformers model name or path")
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.corpus = {}
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

    def get_sentence_embedding_dimension(self):
        """
        Get the dimension of the sentence embeddings.

        Returns
        -------
        int or None
            The dimension of the sentence embeddings, or None if it cannot be determined.
        """
        if hasattr(self.sentence_model, "get_sentence_embedding_dimension"):
            return self.sentence_model.get_sentence_embedding_dimension()
        else:
            return getattr(self.sentence_model.bert.pooler.dense, "out_features", None)

    def add_corpus(self, corpus: Union[List[str], Dict[int, str]], batch_size: int = 32,
                   normalize_embeddings: bool = True):
        """
        Extend the corpus with new documents.
        :param corpus: corpus of documents to use for similarity queries.
        :param batch_size: batch size for computing embeddings
        :param normalize_embeddings: normalize embeddings before computing similarity
        :return: corpus, corpus embeddings
        """
        new_corpus = {}
        start_id = len(self.corpus) if self.corpus else 0
        for id, doc in enumerate(corpus):
            if isinstance(corpus, list):
                if doc not in self.corpus.values():
                    new_corpus[start_id + id] = doc
            else:
                if doc not in self.corpus.values():
                    new_corpus[id] = doc
        self.corpus.update(new_corpus)
        logger.info(f"Start computing corpus embeddings, new docs: {len(new_corpus)}")
        corpus_embeddings = self.get_embeddings(
            list(new_corpus.values()),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        ).tolist()
        if self.corpus_embeddings:
            self.corpus_embeddings = self.corpus_embeddings + corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(new_corpus)} docs, total: {len(self.corpus)}, emb len: {len(self.corpus_embeddings)}")

    def get_embeddings(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = True,
    ):
        """Returns the embeddings for a batch of sentences."""
        return self.sentence_model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]], score_function: str = "cos_sim", **kwargs):
        """
        Compute similarity between two texts.
        :param a: list of str or str
        :param b: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :param kwargs: additional arguments for the similarity function
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        text_emb1 = self.get_embeddings(a, **kwargs)
        text_emb2 = self.get_embeddings(b, **kwargs)

        return score_function(text_emb1, text_emb2)

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(a, b)

    def most_similar(self, queries: Union[str, List[str], Dict[int, str]], topn: int = 10,
                     score_function: str = "cos_sim", **kwargs):
        """
        Find the topn most similar texts to the queries against the corpus.
            It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.
        :param queries: str or list of str
        :param topn: int
        :param score_function: function to compute similarity, default cos_sim
        :param kwargs: additional arguments for the similarity function
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        result = {qid: {} for qid, query in queries.items()}
        queries_ids_map = {i: id for i, id in enumerate(list(queries.keys()))}
        queries_texts = list(queries.values())
        queries_embeddings = self.get_embeddings(queries_texts, convert_to_tensor=True, **kwargs)
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topn, score_function=score_function)
        for idx, hits in enumerate(all_hits):
            for hit in hits[0:topn]:
                result[queries_ids_map[idx]][hit['corpus_id']] = hit['score']

        return result

    def save_corpus_embeddings(self, emb_path: str = "bert_corpus_emb.jsonl"):
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

    def load_corpus_embeddings(self, emb_path: str = "bert_corpus_emb.jsonl"):
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
