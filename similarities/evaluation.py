# -*- coding: utf-8 -*-
"""
refer: https://github.com/UKPLab/beir/blob/main/beir/datasets/data_loader.py
"""
from typing import List, Dict, Tuple

from loguru import logger


def mrr(qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    mrr = {}

    for k in k_values:
        mrr[f"MRR@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logger.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]

    for query_id in qrels:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for rank, hit in enumerate(top_hits[query_id][0:k]):
                if hit[0] in query_relevant_docs:
                    mrr[f"MRR@{k}"] += 1.0 / (rank + 1)
                    break

    for k in k_values:
        mrr[f"MRR@{k}"] = round(mrr[f"MRR@{k}"] / len(qrels), 5)
        logger.info("MRR@{}: {:.4f}".format(k, mrr[f"MRR@{k}"]))

    return mrr


def recall_cap(qrels: Dict[str, Dict[str, int]],
               results: Dict[str, Dict[str, float]],
               k_values: List[int]) -> Tuple[Dict[str, float]]:
    capped_recall = {}

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = 0.0

    k_max = max(k_values)
    logger.info("\n")

    for query_id, doc_scores in results.items():
        top_hits = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        query_relevant_docs = [doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0]
        for k in k_values:
            retrieved_docs = [row[0] for row in top_hits[0:k] if qrels[query_id].get(row[0], 0) > 0]
            denominator = min(len(query_relevant_docs), k)
            capped_recall[f"R_cap@{k}"] += (len(retrieved_docs) / denominator)

    for k in k_values:
        capped_recall[f"R_cap@{k}"] = round(capped_recall[f"R_cap@{k}"] / len(results), 5)
        logger.info("R_cap@{}: {:.4f}".format(k, capped_recall[f"R_cap@{k}"]))

    return capped_recall


def hole(qrels: Dict[str, Dict[str, int]],
         results: Dict[str, Dict[str, float]],
         k_values: List[int]) -> Tuple[Dict[str, float]]:
    hole = {}

    for k in k_values:
        hole[f"Hole@{k}"] = 0.0

    annotated_corpus = set()
    for _, docs in qrels.items():
        for doc_id, score in docs.items():
            annotated_corpus.add(doc_id)

    k_max = max(k_values)
    logger.info("\n")

    for _, scores in results.items():
        top_hits = sorted(scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
        for k in k_values:
            hole_docs = [row[0] for row in top_hits[0:k] if row[0] not in annotated_corpus]
            hole[f"Hole@{k}"] += len(hole_docs) / k

    for k in k_values:
        hole[f"Hole@{k}"] = round(hole[f"Hole@{k}"] / len(results), 5)
        logger.info("Hole@{}: {:.4f}".format(k, hole[f"Hole@{k}"]))

    return hole


def top_k_accuracy(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int]) -> Tuple[Dict[str, float]]:
    top_k_acc = {}

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = 0.0

    k_max, top_hits = max(k_values), {}
    logger.info("\n")

    for query_id, doc_scores in results.items():
        top_hits[query_id] = [item[0] for item in
                              sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]]

    for query_id in qrels:
        query_relevant_docs = set([doc_id for doc_id in qrels[query_id] if qrels[query_id][doc_id] > 0])
        for k in k_values:
            for relevant_doc_id in query_relevant_docs:
                if relevant_doc_id in top_hits[query_id][0:k]:
                    top_k_acc[f"Accuracy@{k}"] += 1.0
                    break

    for k in k_values:
        top_k_acc[f"Accuracy@{k}"] = round(top_k_acc[f"Accuracy@{k}"] / len(qrels), 5)
        logger.info("Accuracy@{}: {:.4f}".format(k, top_k_acc[f"Accuracy@{k}"]))

    return top_k_acc


def evaluate(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int] = (1, 3, 5, 10, 100)) -> Tuple[
    Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    try:
        import pytrec_eval
    except ImportError:
        raise ImportError("Please install pytrec_eval to use this function, eg. `pip install pytrec_eval`")

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)

    for eval in [ndcg, _map, recall, precision]:
        logger.info("\n")
        for k in eval.keys():
            logger.info("{}: {:.4f}".format(k, eval[k]))

    return ndcg, _map, recall, precision


def evaluate_custom(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int] = (1, 3, 5, 10, 100),
        metric: str = 'acc') -> Tuple[Dict[str, float]]:
    if metric.lower() in ["mrr", "mrr@k", "mrr_cut"]:
        return mrr(qrels, results, k_values)

    elif metric.lower() in ["recall_cap", "r_cap", "r_cap@k"]:
        return recall_cap(qrels, results, k_values)

    elif metric.lower() in ["hole", "hole@k"]:
        return hole(qrels, results, k_values)

    elif metric.lower() in ["acc", "top_k_acc", "accuracy", "accuracy@k", "top_k_accuracy"]:
        return top_k_accuracy(qrels, results, k_values)
