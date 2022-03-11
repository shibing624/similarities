# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import datetime
import os
import pathlib
import random
import sys
import numpy as np
from loguru import logger

sys.path.append('../..')
from similarities import Similarity
from similarities.utils import http_get
from similarities.data_loader import SearchDataLoader
from similarities.evaluation import evaluate

random.seed(42)

pwd_path = os.path.dirname(os.path.realpath(__file__))


#### Download scifact.zip dataset and unzip the dataset
def get_scifact():
    # Download scifact.zip dataset and unzip the dataset
    dataset = "scifact"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    zip_file = os.path.join(pwd_path, "scifact.zip")
    if not os.path.exists(zip_file):
        logger.info("Dataset not exists, downloading...")
        http_get(url, zip_file, extract=True)
    else:
        logger.info("Dataset already exists, skipping download.")
    data_path = os.path.join(pwd_path, dataset)
    return data_path


def get_dbpedia():
    dataset = "dbpedia-entity"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    zip_file = os.path.join(pwd_path, "dbpedia-entity.zip")
    if not os.path.exists(zip_file):
        logger.info("Dataset not exists, downloading...")
        http_get(url, zip_file, extract=True)
    else:
        logger.info("Dataset already exists, skipping download.")
    data_path = os.path.join(pwd_path, dataset)
    return data_path


data_path = get_scifact()
#### Loading test queries and corpus in DBPedia
corpus, queries, qrels = SearchDataLoader(data_path).load(split="test")
corpus_ids, query_ids = list(corpus), list(queries)
logger.info(f"corpus: {len(corpus)}, queries: {len(queries)}")

# query_keys = list(queries.keys())[:10]
# queries = {key: queries[key] for key in query_keys}
# print(len(queries))
# print(len(qrels))

#### Randomly sample 1M pairs from Original Corpus (4.63M pairs)
#### First include all relevant documents (i.e. present in qrels)
corpus_set = set()
for query_id in qrels:
    corpus_set.update(list(qrels[query_id].keys()))
corpus_new = {corpus_id: corpus[corpus_id] for corpus_id in corpus_set}

#### Remove already seen k relevant documents and sample (1M - k) docs randomly
remaining_corpus = list(set(corpus_ids) - corpus_set)
sample = min(1000000 - len(corpus_set), len(remaining_corpus))

for corpus_id in random.sample(remaining_corpus, sample):
    corpus_new[corpus_id] = corpus[corpus_id]

corpus_docs = {corpus_id: corpus_new[corpus_id]['title'] + corpus_new[corpus_id]['text'] for corpus_id, corpus in
               corpus_new.items()}
#### Index 1M passages into the index (seperately)
model = Similarity(corpus=corpus_docs, model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
logger.debug(model)
#### Saving benchmark times with batch
# queries = [queries[query_id] for query_id in query_ids]
start = datetime.datetime.now()
results = model.most_similar(queries, topn=10)
end = datetime.datetime.now()
#### Measuring time taken in ms (milliseconds)
time_taken = (end - start)
time_taken = time_taken.total_seconds() * 1000
logger.info("All, Spend {:.2f}ms".format(time_taken))
logger.info("Average time taken: {:.2f}ms".format(time_taken / len(queries)))
logger.info(f"Results size: {len(results)}")

#### Evaluate your retrieval using NDCG@k, MAP@K ...
ndcg, _map, recall, precision = evaluate(qrels, results)
logger.info(f"MAP: {_map}")

#### Measuring Index size consumed by document embeddings
corpus_embs = model.corpus_embeddings
cpu_memory = sys.getsizeof(np.array(corpus_embs, dtype=np.float32))

logger.info("Number of documents: {}, Dim: {}".format(len(corpus_embs), len(corpus_embs[0])))
logger.info("Index size (in MB): {:.2f}MB".format(cpu_memory * 0.000001))
