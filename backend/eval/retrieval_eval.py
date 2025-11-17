import numpy as np

def precision_at_k(retrieved_ids, gold_ids, k=5):
    retrieved_k = retrieved_ids[:k]
    correct = sum(1 for r in retrieved_k if r in gold_ids)
    return correct / k

def recall_at_k(retrieved_ids, gold_ids, k=5):
    retrieved_k = retrieved_ids[:k]
    correct = sum(1 for r in retrieved_k if r in gold_ids)
    return correct / len(gold_ids) if gold_ids else 0.0

def f1_at_k(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def mrr(retrieved_ids, gold_ids):
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in gold_ids:
            return 1.0 / rank
    return 0.0

def ndcg(retrieved_ids, gold_ids, k=5):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gold_ids:
            dcg += 1 / np.log2(i + 1)

    ideal_dcg = 1 / np.log2(1 + 1)  # best case: hit at rank 1

    return dcg / (ideal_dcg + 1e-9)

def evaluate_retrieval(gold_ids, retrieved_docs, k=5):
    retrieved_ids = [d["doc_id"] for d in retrieved_docs]

    p = precision_at_k(retrieved_ids, gold_ids, k)
    r = recall_at_k(retrieved_ids, gold_ids, k)
    f = f1_at_k(p, r)
    m = mrr(retrieved_ids, gold_ids)
    n = ndcg(retrieved_ids, gold_ids, k)

    return {
        "precision@k": p,
        "recall@k": r,
        "f1@k": f,
        "mrr": m,
        "ndcg": n
    }
