import numpy as np


def compute_map_at_k(dataset, k=5):
    """
    Computes Mean Average Precision (MAP@k).

    Parameters:
    - dataset: List of query-retrieved results and ground truth relevance.
    - k: Number of top results to consider.

    Returns:
    - MAP@k score.
    """
    ap_scores = []
    for entry in dataset:
        retrieved = entry["retrieved_docs"]
        relevant = entry["pos_chunk"]

        ap = 0
        for rank, doc in enumerate(retrieved, 1):
            if doc == relevant:
                ap = 1 / rank
                break
        ap_scores.append(ap)

    return np.mean(ap_scores)


def compute_success_at_k(dataset, k=5):
    """
    Computes Success@k for single relevant document per query.

    Parameters:
    - dataset: List of queries with retrieved results and a single relevant document.
    - k: Number of top results to consider.

    Returns:
    - Success@k score.
    """
    success_count = 0

    for entry in dataset:
        retrieved = entry["retrieved_docs"]
        relevant = entry["pos_chunk"]

        if relevant in retrieved:
            success_count += 1

    return success_count / len(dataset)  # Proportion of successful queries
