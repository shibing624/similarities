# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import os
from typing import List

import numpy as np
import torch
from loguru import logger
from text2vec import SentenceModel

from similarities.similarity import cos_sim, semantic_search, dot_score

pwd_path = os.path.abspath(os.path.dirname(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BertSimilarity(object):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar terms for a given term.
    """

    def __init__(self, sentencemodel: SentenceModel, docs: List[str] = None):
        # super().__init__()
        self.sentencemodel = sentencemodel
        self.docs = []
        self.docs_embeddings = np.array([])
        if docs is not None:
            self.add_documents(docs)

    def __len__(self):
        """Get length of docs."""
        return self.docs_embeddings.shape[0]

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def add_documents(self, docs):
        """
        Extend the docs_embeddings with new documents.

        Parameters
        ----------
        docs : list of str
        """
        self.docs += docs
        docs_embeddings = self.get_vector(docs)
        if self.docs_embeddings.size > 0:
            self.docs_embeddings = np.vstack((self.docs_embeddings, docs_embeddings))
        else:
            self.docs_embeddings = docs_embeddings
        logger.info(f"Add docs size: {len(docs)}, total size: {len(self.docs)}")

    def get_vector(self, text):
        return self.sentencemodel.encode(text)

    def similarity(self, text1, text2, score_function=cos_sim):
        text_emb1 = self.get_vector(text1)
        text_emb2 = self.get_vector(text2)
        return score_function(text_emb1, text_emb2)

    def distance(self, text1, text2):
        """Compute cosine distance between two keys.
        """
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query, topn=10):
        result = []
        query_embeddings = self.get_vector(query)
        hits = semantic_search(query_embeddings, self.docs_embeddings, top_k=topn)
        hits = hits[0]  # Get the hits for the first query

        print("Input question:", query)
        for hit in hits[0:topn]:
            result.append((self.docs[hit['corpus_id']], round(hit['score'], 4)))
            print("\t{:.3f}\t{}".format(hit['score'], self.docs[hit['corpus_id']]))

        print("\n\n========\n")
        return result



class AnnoySimilarity(object):
    """
    Computes cosine similarities between word embeddings and retrieves most
    similar terms for a given term.
    """

    def __init__(self, sentencemodel: SentenceModel, docs: List[str] = None):
        # super().__init__()
        self.sentencemodel = sentencemodel
        self.docs = []
        self.docs_embeddings = np.array([])
        if docs is not None:
            self.add_documents(docs)

    def __len__(self):
        """Get length of docs."""
        return self.docs_embeddings.shape[0]

    def __str__(self):
        return "%s" % (self.__class__.__name__)

    def add_documents(self, docs):
        """
        Extend the docs_embeddings with new documents.

        Parameters
        ----------
        docs : list of str
        """
        self.docs += docs
        docs_embeddings = self.get_vector(docs)
        if self.docs_embeddings.size > 0:
            self.docs_embeddings = np.vstack((self.docs_embeddings, docs_embeddings))
        else:
            self.docs_embeddings = docs_embeddings
        logger.info(f"Add docs size: {len(docs)}, total size: {len(self.docs)}")

    def get_vector(self, text):
        return self.sentencemodel.encode(text)

    def similarity(self, text1, text2, score_function=cos_sim):
        text_emb1 = self.get_vector(text1)
        text_emb2 = self.get_vector(text2)
        return score_function(text_emb1, text_emb2)

    def distance(self, text1, text2):
        """Compute cosine distance between two keys.
        """
        return 1 - self.similarity(text1, text2)

    def most_similar(self, query, topn=10):
        result = []
        query_embeddings = self.get_vector(query)
        hits = semantic_search(query_embeddings, self.docs_embeddings, top_k=topn)
        hits = hits[0]  # Get the hits for the first query

        print("Input question:", query)
        for hit in hits[0:topn]:
            result.append((self.docs[hit['corpus_id']], round(hit['score'], 4)))
            print("\t{:.3f}\t{}".format(hit['score'], self.docs[hit['corpus_id']]))

        print("\n\n========\n")
        return result


if __name__ == '__main__':
    sm = SentenceModel()
    list_of_docs = ["This is a test1", "This is a test2", "This is a test3", '刘若英是个演员', '他唱歌很好听', 'women喜欢这首歌']
    list_of_docs2 = ["that is test4", "that is a test5", "that is a test6", '刘若英个演员', '唱歌很好听', 'men喜欢这首歌']
    m = BertSimilarity(sm, list_of_docs)
    m.add_documents(list_of_docs2)
    v = m.get_vector("This is a test1")
    print(v[:10], v.shape)
    print(m.similarity("This is a test1", "that is a test5"))
    print(m.distance("This is a test1", "that is a test5"))
    print(m.most_similar("This is a test1"))
    print(m.most_similar("这是个演员"))
