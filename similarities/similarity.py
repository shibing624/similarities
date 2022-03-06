# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import queue
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional
from loguru import logger
from text2vec import SentenceModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cos_sim(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_score(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def pairwise_dot_score(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
    Computes the pairwise dot-product dot_prod(a[i], b[i])
    :return: Vector with res[i] = dot_prod(a[i], b[i])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return (a * b).sum(dim=-1)


def pairwise_cos_sim(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
   Computes the pairwise cossim cos_sim(a[i], b[i])
   :return: Vector with res[i] = cos_sim(a[i], b[i])
   """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    return pairwise_dot_score(normalize_embeddings(a), normalize_embeddings(b))


def normalize_embeddings(embeddings: torch.Tensor):
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def semantic_search(
        query_embeddings: Union[torch.Tensor, np.ndarray],
        corpus_embeddings: Union[torch.Tensor, np.ndarray],
        query_chunk_size: int = 100,
        corpus_chunk_size: int = 500000,
        top_k: int = 10,
        score_function=cos_sim
):
    """
    This function performs a cosine similarity search between a list of query embeddings and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.

    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but
        requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed,
        but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Funtion for computing scores. By default, cosine similarity.
    :return: Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the
        keys 'corpus_id' and 'score'
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # Check that corpus and queries are on the same device
    query_embeddings = query_embeddings.to(device)
    corpus_embeddings = corpus_embeddings.to(device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # Iterate over chunks of the corpus
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # Compute cosine similarity
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                        corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

            # Get top-k scores
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])),
                                                                       dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr],
                                                cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

    # Sort and strip to top_k results
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

    return queries_result_list


def paraphrase_mining_embeddings(
        embeddings: Union[torch.Tensor, np.ndarray],
        query_chunk_size: int = 5000,
        corpus_chunk_size: int = 100000,
        max_pairs: int = 500000,
        top_k: int = 100,
        score_function=cos_sim
):
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param embeddings: A tensor with the embeddings
    :param query_chunk_size: Search for most similar pairs for #query_chunk_size at the same time. Decrease, to lower
        memory footprint (increases run-time).
    :param corpus_chunk_size: Compare a sentence simultaneously against #corpus_chunk_size other sentences. Decrease,
        to lower memory footprint (increases run-time).
    :param max_pairs: Maximal number of text pairs returned.
    :param top_k: For each sentence, we retrieve up to top_k other sentences
    :param score_function: Function for computing scores. By default, cosine similarity.
    :return: Returns a list of triplets with the format [score, id1, id2]
    """
    if isinstance(embeddings, (np.ndarray, np.generic)):
        embeddings = torch.from_numpy(embeddings)
    elif isinstance(embeddings, list):
        embeddings = torch.stack(embeddings)

    if len(embeddings.shape) == 1:
        embeddings = embeddings.unsqueeze(0)
    embeddings = embeddings.to(device)

    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            scores = score_function(embeddings[query_start_idx: query_start_idx + query_chunk_size],
                                    embeddings[corpus_start_idx: corpus_start_idx + corpus_chunk_size])

            scores_top_k_values, scores_top_k_idx = torch.topk(scores, min(top_k, len(scores[0])), dim=1, largest=True,
                                                               sorted=False)
            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(scores)):
                for top_k_idx, corpus_itr in enumerate(scores_top_k_idx[query_itr]):
                    i = query_start_idx + query_itr
                    j = corpus_start_idx + corpus_itr

                    if i != j and scores_top_k_values[query_itr][top_k_idx] > min_score:
                        pairs.put((scores_top_k_values[query_itr][top_k_idx], i, j))
                        num_added += 1

                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])

        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, i, j])

    # Highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list


def community_detection(embeddings, threshold=0.75, min_community_size=10, init_max_size=1000):
    """
    Function for Fast Community Detection

    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).

    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """

    # Maximum size for community
    init_max_size = min(init_max_size, len(embeddings))

    # Compute cosine similarity scores
    cos_scores = cos_sim(embeddings, embeddings)

    # Minimum size for a community
    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities


class Similarity:
    """
    Compute similarity:
    1. Compute the similarity between two sentences
    2. Retrieves most similar sentence of a query against a corpus of documents.

    The index supports adding new documents dynamically.
    """

    def __init__(self, sentence_model: Union[str, SentenceModel], corpus: List[str] = None):
        """
        Initialize the similarity object.
        :param sentence_model: Model to use for sentence embeddings.
        :param corpus: Corpus of documents to use for similarity queries.
        """
        if isinstance(sentence_model, SentenceModel):
            self.sentence_model = sentence_model
        elif isinstance(sentence_model, str):
            self.sentence_model = SentenceModel(sentence_model)
        else:
            raise ValueError("sentence_model must be either a SentenceModel or a model name of SentenceTransformer.")
        self.corpus = []
        self.corpus_embeddings = np.array([])
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
        docs_embeddings = self.get_vector(corpus)
        if self.corpus_embeddings.size > 0:
            self.corpus_embeddings = np.vstack((self.corpus_embeddings, docs_embeddings))
        else:
            self.corpus_embeddings = docs_embeddings
        logger.info(f"Add docs size: {len(corpus)}, total size: {len(self.corpus)}")

    def get_vector(self, text: Union[str, List[str]]):
        """
        Returns the embeddings for a batch of sentences.
        :param text:
        :return:
        """
        return self.sentence_model.encode(text)

    def similarity(self, text1: Union[str, List[str]], text2: Union[str, List[str]], score_function=cos_sim):
        """
        Compute similarity between two texts.
        :param text1: list of str or str
        :param text2: list of str or str
        :param score_function: function to compute similarity, default cos_sim
        :return: similarity score, torch.Tensor, Matrix with res[i][j] = cos_sim(a[i], b[j])
        """
        text_emb1 = self.get_vector(text1)
        text_emb2 = self.get_vector(text2)
        return score_function(text_emb1, text_emb2)

    def distance(self, text1: Union[str, List[str]], text2: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query: str, topn: int = 10):
        """
        Find the topn most similar texts to the query against the corpus.
        :param query: str
        :param topn: int
        :return:
        """
        result = []
        query_embeddings = self.get_vector(query)
        hits = semantic_search(query_embeddings, self.corpus_embeddings, top_k=topn)
        hits = hits[0]  # Get the first query result when query is string

        for hit in hits[0:topn]:
            result.append((hit['corpus_id'], self.corpus[hit['corpus_id']], hit['score']))

        return result
