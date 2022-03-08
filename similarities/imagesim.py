# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Image similarity and image retrieval

refer: https://colab.research.google.com/drive/1leOzG-AQw5MkzgA4qNW5fb3yc-oJ4Lo4
Adjust the code to compare similarity score and search.
"""
import math
from typing import List, Union

import numpy as np
from PIL import Image
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from similarities.similarity import semantic_search
from similarities.utils.distance import hamming_distance
from similarities.utils.imagehash import phash, dhash, whash, average_hash
from similarities.utils.util import cos_sim


class ClipSimilarity:
    """
    Compute CLIP similarity between two images and retrieves most
    similar image for a given image corpus.

    CLIP: https://github.com/openai/CLIP.git
    """

    def __init__(self, corpus: List[str] = None, model_name_or_path: str = 'clip-ViT-B-32'):
        self.corpus = []
        self.clip_model = SentenceTransformer(model_name_or_path)  # load the CLIP model
        self.corpus_embeddings = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """Get length of corpus."""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: CLIP"
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
        corpus_embeddings = self._get_vector(corpus).tolist()
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add corpus size: {len(corpus)}, total size: {len(self.corpus)}")

    def _convert_to_rgb(self, img):
        """Convert image to RGB mode."""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def _get_vector(self, img_paths: Union[str, List[str]]):
        """
        Returns the embeddings for a batch of images.
        :param img_paths:
        :return:
        """
        if isinstance(img_paths, str):
            img_paths = [img_paths]
        imgs = [Image.open(filepath) for filepath in img_paths]
        imgs = [self._convert_to_rgb(img) for img in imgs]
        return self.clip_model.encode(imgs, batch_size=128, convert_to_tensor=False, show_progress_bar=True)

    def similarity(self, img_paths1: Union[str, List[str]], img_paths2: Union[str, List[str]]):
        """
        Compute similarity between two image files.
        :param img_paths1: image file path 1
        :param img_paths2: image file path 2
        :return: similarity score
        """
        embs1 = self._get_vector(img_paths1)
        embs2 = self._get_vector(img_paths2)
        similarity_score = cos_sim(embs1, embs2)

        return similarity_score

    def distance(self, fp1: str, fp2: str):
        """Compute distance between two image files."""
        return 1 - self.similarity(fp1, fp2)

    def most_similar(self, query_fp: str, topn: int = 10):
        """
        Find the topn most similar images to the query against the corpus.
        :param query_fp: str
        :param topn: int
        :return: list of tuples (id, image_path, similarity)
        """
        result = []
        q_emb = self._get_vector(query_fp)

        # Computes the cosine-similarity between the query embedding and all image embeddings.
        hits = semantic_search(q_emb, np.array(self.corpus_embeddings, dtype=np.float32), top_k=topn)
        hits = hits[0]  # Get the first query result when query is string

        for hit in hits[:topn]:
            result.append((hit['corpus_id'], self.corpus[hit['corpus_id']], hit['score']))
        return result


class ImageHashSimilarity:
    """
    Compute Phash similarity between two images and retrieves most
    similar image for a given image corpus.

    perceptual hash (pHash), which acts as an image fingerprint.
    """

    def __init__(self, corpus: List[str] = None, hash_function: str = "phash", hash_size: int = 16):
        self.corpus = []
        self.hash_functions = {'phash': phash, 'dhash': dhash, 'whash': whash, 'average_hash': average_hash}
        if hash_function not in self.hash_functions:
            raise ValueError(f"hash_function: {hash_function} must be one of {self.hash_functions.keys()}")
        self.hash_function = self.hash_functions[hash_function]
        self.hash_size = hash_size
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

    def add_corpus(self, corpus: List[str]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        self.corpus += corpus
        corpus_embeddings = []
        for doc_fp in tqdm(corpus, desc="Calculating corpus image hash"):
            doc_seq = str(self.hash_function(Image.open(doc_fp), self.hash_size))
            corpus_embeddings.append(doc_seq)
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add corpus size: {len(corpus)}, total size: {len(self.corpus)}")

    def _sim_score(self, seq1, seq2):
        """Compute hamming similarity between two seqs."""
        return 1.0 - hamming_distance(seq1, seq2) / len(seq1)

    def similarity(self, fp1: str, fp2: str):
        """
        Compute similarity between two image files.
        :param fp1: image file path 1
        :param fp2: image file path 2
        :return: similarity score
        """
        img1 = Image.open(fp1)
        img2 = Image.open(fp2)
        seq1 = str(self.hash_function(img1, self.hash_size))
        seq2 = str(self.hash_function(img2, self.hash_size))
        similarity_score = self._sim_score(seq1, seq2)

        return similarity_score

    def distance(self, fp1: str, fp2: str):
        """Compute distance between two image files."""
        return 1 - self.similarity(fp1, fp2)

    def most_similar(self, query_fp: str, topn: int = 10):
        """
        Find the topn most similar images to the query against the corpus.
        :param query_fp: str
        :param topn: int
        :return: list of tuples (id, image_path, similarity)
        """
        result = []
        q_seq = str(self.hash_function(Image.open(query_fp), self.hash_size))
        for (corpus_id, doc), doc_seq in zip(enumerate(self.corpus), self.corpus_embeddings):
            score = self._sim_score(q_seq, doc_seq)
            result.append((corpus_id, doc, score))
        result.sort(key=lambda x: x[2], reverse=True)
        return result[:topn]


class SiftSimilarity:
    """
    Compute SIFT similarity between two images and retrieves most
    similar image for a given image corpus.

    SIFT, Scale Invariant Feature Transform(SIFT) 尺度不变特征变换匹配算法详解
    https://blog.csdn.net/zddblog/article/details/7521424
    """

    def __init__(self, corpus: List[str] = None, nfeatures: int = 500):
        try:
            import cv2
        except ImportError:
            raise ImportError("Install cv2 to use SiftSimilarity, e.g. `pip install cv2`")
        self.corpus = []
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

    def add_corpus(self, corpus: List[str]):
        """
        Extend the corpus with new documents.

        Parameters
        ----------
        corpus : list of str
        """
        self.corpus += corpus
        corpus_embeddings = []
        for doc_fp in tqdm(corpus, desc="Calculating corpus image SIFT"):
            img = Image.open(doc_fp)
            _, descriptors = self.calculate_descr(img)
            if len(descriptors.shape) > 0 and descriptors.shape[0] > 0:
                corpus_embeddings.append(descriptors.tolist())
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add corpus size: {len(corpus)}, total size: {len(self.corpus)}")

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
        if len(good_matches) < 5:
            return score
        bestN = 5
        topBestNSum = 0
        good_matches.sort(key=lambda match: match.distance)
        for match in good_matches[:bestN]:
            topBestNSum += match.distance
        score = (topBestNSum / bestN) * good_matches_sum / len(good_matches)
        return score

    def similarity(self, fp1: str, fp2: str):
        """
        Compute similarity between two image files.
        :param fp1: image file path 1
        :param fp2: image file path 2
        :return: similarity score
        """
        similarity_score = 0.0
        _, desc1 = self.calculate_descr(Image.open(fp1))
        _, desc2 = self.calculate_descr(Image.open(fp2))
        if desc1.size > 0 and desc2.size > 0:
            similarity_score = self._sim_score(desc1, desc2)

        return similarity_score

    def distance(self, fp1: str, fp2: str):
        """Compute distance between two keys."""
        return 1 - self.similarity(fp1, fp2)

    def most_similar(self, query_fp: str, topn: int = 10):
        """
        Find the topn most similar images to the query against the corpus.
        :param query_fp: str
        :param topn: int
        :return: list of tuples (id, image_path, similarity)
        """
        result = []
        _, q_desc = self.calculate_descr(Image.open(query_fp))
        for (corpus_id, doc), doc_desc in zip(enumerate(self.corpus), self.corpus_embeddings):
            score = self._sim_score(q_desc, doc_desc)
            result.append((corpus_id, doc, score))
        result.sort(key=lambda x: x[2], reverse=True)
        return result[:topn]
