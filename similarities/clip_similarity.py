# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Image similarity and image retrieval

refer: https://colab.research.google.com/drive/1leOzG-AQw5MkzgA4qNW5fb3yc-oJ4Lo4
Adjust the code to compare similarity score and search.
"""

from typing import List, Union, Dict
import json
import numpy as np
from PIL import Image
from loguru import logger

from similarities.clip_module import ClipModule
from similarities.similarity import SimilarityABC
from similarities.utils.util import cos_sim, semantic_search, dot_score


class ClipSimilarity(SimilarityABC):
    """
    Compute CLIP similarity between two images and retrieves most
    similar image for a given image corpus.

    CLIP: https://github.com/openai/CLIP.git
    english model: openai/clip-vit-base-patch32
    chinese model: OFA-Sys/chinese-clip-vit-base-patch16
    """

    def __init__(
            self,
            corpus: Union[List[Image.Image], Dict[str, Image.Image]] = None,
            model_name_or_path: str ='OFA-Sys/chinese-clip-vit-base-patch16',
            device: str = None,
    ):
        self.clip_model = ClipModule(model_name_or_path, device=device)  # load the CLIP model
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.corpus = {}
        self.corpus_embeddings = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.clip_model.__class__.__name__}"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def _convert_to_rgb(self, img):
        """Convert image to RGB mode."""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def get_embeddings(
            self,
            text_or_img: Union[List[Image.Image], Image.Image, str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = True,
    ):
        """
        Returns the embeddings for a batch of images.
        :param text_or_img: list of str or Image.Image or image list
        :param batch_size: batch size
        :param show_progress_bar: show progress bar
        :param convert_to_numpy: convert embeddings to numpy
        :param convert_to_tensor: convert embeddings to tensor
        :param device: device
        :param normalize_embeddings: normalize embeddings
        :return: np.ndarray, embeddings for the given images
        """
        if isinstance(text_or_img, str):
            text_or_img = [text_or_img]
        if isinstance(text_or_img, Image.Image):
            text_or_img = [text_or_img]
        if isinstance(text_or_img, list) and isinstance(text_or_img[0], Image.Image):
            text_or_img = [self._convert_to_rgb(i) for i in text_or_img]
        return self.clip_model.encode(
            text_or_img,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

    def add_corpus(self, corpus: Union[List[Image.Image], Dict[str, Image.Image]], batch_size: int = 32,
                   normalize_embeddings: bool = True):
        """Extend the corpus with new documents."""
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)
        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_embeddings = self.get_embeddings(
            list(corpus_new.values()),
            show_progress_bar=True,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        ).tolist()
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    def similarity(
            self,
            a: Union[List[Image.Image], Image.Image, str, List[str]],
            b: Union[List[Image.Image], Image.Image, str, List[str]],
            score_function: str = "cos_sim",
            **kwargs
    ):
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

    def most_similar(self, queries, topn: int = 10, **kwargs):
        """
        Find the topn most similar texts to the queries against the corpus.
        :param queries: text or image
        :param topn: int
        :param kwargs: other params
        :return: Dict[str, Dict[str, float]], {query_id: {corpus_id: similarity_score}, ...}
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_ids_map = {i: id for i, id in enumerate(list(queries.keys()))}
        queries_texts = list(queries.values())
        queries_embeddings = self.get_embeddings(queries_texts, **kwargs)
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topn)
        for idx, hits in enumerate(all_hits):
            for hit in hits[0:topn]:
                result[queries_ids_map[idx]][hit['corpus_id']] = hit['score']

        return result

    def save_corpus_embeddings(self, emb_path: str = "clip_corpus_emb.jsonl"):
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

    def load_corpus_embeddings(self, emb_path: str = "clip_corpus_emb.jsonl"):
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
