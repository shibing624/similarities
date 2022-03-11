# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Image similarity and image retrieval

refer: https://colab.research.google.com/drive/1leOzG-AQw5MkzgA4qNW5fb3yc-oJ4Lo4
Adjust the code to compare similarity score and search.
"""

from typing import List, Union, Dict

import math
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

from similarities.similarity import SimilarityABC, Similarity
from similarities.utils.distance import hamming_distance
from similarities.utils.imagehash import phash, dhash, whash, average_hash
from similarities.clip_model import CLIPModel


class ClipSimilarity(Similarity):
    """
    Compute CLIP similarity between two images and retrieves most
    similar image for a given image corpus.

    CLIP: https://github.com/openai/CLIP.git
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[str, str]] = None,
            model_name_or_path='openai/clip-vit-base-patch32'
    ):
        self.clip_model = CLIPModel(model_name_or_path)  # load the CLIP model
        super().__init__(corpus, self.clip_model)

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

    def _get_vector(self, img_paths: Union[str, List[str]], show_progress_bar: bool = False):
        """
        Returns the embeddings for a batch of images.
        :param img_paths:
        :return:
        """
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        imgs = [Image.open(filepath) for filepath in img_paths]
        imgs = [self._convert_to_rgb(img) for img in imgs]
        return self.clip_model.encode(imgs, batch_size=128, show_progress_bar=show_progress_bar)


class ImageHashSimilarity(SimilarityABC):
    """
    Compute Phash similarity between two images and retrieves most
    similar image for a given image corpus.

    perceptual hash (pHash), which acts as an image fingerprint.
    """

    def __init__(self, corpus: Union[List[str], Dict[str, str]] = None,
                 hash_function: str = "phash", hash_size: int = 16):
        self.hash_functions = {'phash': phash, 'dhash': dhash, 'whash': whash, 'average_hash': average_hash}
        if hash_function not in self.hash_functions:
            raise ValueError(f"hash_function: {hash_function} must be one of {self.hash_functions.keys()}")
        self.hash_function = self.hash_functions[hash_function]
        self.hash_size = hash_size
        self.corpus = {}
        self.corpus_embeddings = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.hash_function.__name__}"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
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
        self.corpus_ids_map = {i: id for i, id in enumerate(list(self.corpus.keys()))}
        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_embeddings = []
        for doc_fp in tqdm(list(corpus_new.values()), desc="Calculating corpus image hash"):
            doc_seq = str(self.hash_function(Image.open(doc_fp), self.hash_size))
            corpus_embeddings.append(doc_seq)
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    def _sim_score(self, seq1, seq2):
        """Compute hamming similarity between two seqs."""
        return 1.0 - hamming_distance(seq1, seq2) / len(seq1)

    def similarity(self, img_paths1: Union[str, List[str]], img_paths2: Union[str, List[str]]):
        """
        Compute similarity between two image files.
        :param img_paths1: image file paths 1
        :param img_paths2: image file paths 2
        :return: list of float, similarity score
        """
        if isinstance(img_paths1, str):
            img_paths1 = [img_paths1]
        if isinstance(img_paths2, str):
            img_paths2 = [img_paths2]
        if len(img_paths1) != len(img_paths2):
            raise ValueError("expected two inputs of the same length")

        seqs1 = [str(self.hash_function(Image.open(i), self.hash_size)) for i in img_paths1]
        seqs2 = [str(self.hash_function(Image.open(i), self.hash_size)) for i in img_paths2]
        scores = [self._sim_score(seq1, seq2) for seq1, seq2 in zip(seqs1, seqs2)]
        return scores

    def distance(self, img_paths1: Union[str, List[str]], img_paths2: Union[str, List[str]]):
        """Compute distance between two image files."""
        sim_scores = self.similarity(img_paths1, img_paths2)
        return [1.0 - score for score in sim_scores]

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """
        Find the topn most similar images to the query against the corpus.
        :param queries: str of list of str, image file paths
        :param topn: int
        :return: list of list tuples (id, image_path, similarity)
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            q_seq = str(self.hash_function(Image.open(query), self.hash_size))
            for (corpus_id, doc), doc_seq in zip(self.corpus.items(), self.corpus_embeddings):
                score = self._sim_score(q_seq, doc_seq)
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score
        return result


class SiftSimilarity(SimilarityABC):
    """
    Compute SIFT similarity between two images and retrieves most
    similar image for a given image corpus.

    SIFT, Scale Invariant Feature Transform(SIFT) 尺度不变特征变换匹配算法详解
    https://blog.csdn.net/zddblog/article/details/7521424
    """

    def __init__(self, corpus: Union[List[str], Dict[str, str]] = None, nfeatures: int = 500):
        try:
            import cv2
        except ImportError:
            raise ImportError("Install cv2 to use SiftSimilarity, e.g. `pip install opencv-python`")
        self.corpus = {}
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)
        self.bf_matcher = cv2.BFMatcher()  # Brute-force matcher create method.
        self.corpus_embeddings = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: SIFT"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
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
        self.corpus_ids_map = {i: id for i, id in enumerate(list(self.corpus.keys()))}
        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_embeddings = []
        for doc_fp in tqdm(list(corpus_new.values()), desc="Calculating corpus image SIFT"):
            img = Image.open(doc_fp)
            _, descriptors = self.calculate_descr(img)
            if len(descriptors.shape) > 0 and descriptors.shape[0] > 0:
                corpus_embeddings.append(descriptors.tolist())
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    @staticmethod
    def _resize_img_to_array(img, max_height=2000, max_width=2000):
        """Resize image to array."""
        height, width = img.size
        if height * width > max_height * max_width:
            k = math.sqrt(height * width / (max_height * max_width))
            img = img.resize(
                (round(height / k), round(width / k)),
                Image.ANTIALIAS
            )
        img_array = np.array(img)
        return img_array

    def calculate_descr(self, img, min_value=1e-7):
        """Calculate SIFT descriptors."""
        img = self._resize_img_to_array(img)
        key_points, descriptors = self.sift.detectAndCompute(img, None)
        if descriptors is None:
            return None, None
        descriptors /= (descriptors.sum(axis=1, keepdims=True) + min_value)  # RootSift
        descriptors = np.sqrt(descriptors)
        return key_points, descriptors

    def _sim_score(self, desc1, desc2):
        """Compute similarity between two descs."""
        if isinstance(desc1, list):
            desc1 = np.array(desc1, dtype=np.float32)
        if isinstance(desc2, list):
            desc2 = np.array(desc2, dtype=np.float32)
        score = 0.0
        matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []
        good_matches_sum = 0
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                good_matches_sum += m.distance
        if len(good_matches) < 3:
            return score
        bestN = 5
        topBestNSum = 0
        good_matches.sort(key=lambda match: match.distance)
        for match in good_matches[:bestN]:
            topBestNSum += match.distance
        score = (topBestNSum / bestN) * good_matches_sum / len(good_matches)
        return score

    def similarity(self, img_paths1: Union[str, List[str]], img_paths2: Union[str, List[str]]):
        """
        Compute similarity between two image files.
        :param img_paths1: image file paths 1
        :param img_paths2: image file paths 2
        :return: list of float, similarity score
        """
        if isinstance(img_paths1, str):
            img_paths1 = [img_paths1]
        if isinstance(img_paths2, str):
            img_paths2 = [img_paths2]
        if len(img_paths1) != len(img_paths2):
            raise ValueError("expected two inputs of the same length")

        scores = []
        for fp1, fp2 in zip(img_paths1, img_paths2):
            score = 0.0
            _, desc1 = self.calculate_descr(Image.open(fp1))
            _, desc2 = self.calculate_descr(Image.open(fp2))
            if desc1.size > 0 and desc2.size > 0:
                score = self._sim_score(desc1, desc2)
            scores.append(score)

        return scores

    def distance(self, img_paths1: Union[str, List[str]], img_paths2: Union[str, List[str]]):
        """Compute distance between two keys."""
        sim_scores = self.similarity(img_paths1, img_paths2)
        return [1.0 - score for score in sim_scores]

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """
        Find the topn most similar images to the query against the corpus.
        :param queries: str of list of str, image file paths
        :param topn: int
        :return: list of list tuples (id, image_path, similarity)
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}

        for qid, query in queries.items():
            q_res = []
            _, q_desc = self.calculate_descr(Image.open(query))
            for (corpus_id, doc), doc_desc in zip(enumerate(self.corpus), self.corpus_embeddings):
                score = self._sim_score(q_desc, doc_desc)
                q_res.append((corpus_id, score))
            q_res.sort(key=lambda x: x[1], reverse=True)
            q_res = q_res[:topn]
            for corpus_id, score in q_res:
                result[qid][corpus_id] = score

        return result
