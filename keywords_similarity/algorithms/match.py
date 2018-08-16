import numpy as np


def _greedy_matching_similarity(similarity_matrix):
    n_1, n_2 = similarity_matrix.shape  # n_1 <= n_2

    total_score = 0.

    matched_1 = set()
    matched_2 = set()

    sorted_ixs = np.unravel_index(
        similarity_matrix.argsort(axis=None)[:: -1],
        similarity_matrix.shape,
    )

    for i_1, i_2 in zip(*sorted_ixs):
        if i_1 in matched_1 or i_2 in matched_2:
            continue

        total_score += similarity_matrix[i_1, i_2]
        matched_1.add(i_1)
        matched_2.add(i_2)

        if len(matched_1) == n_1:
            break

    for i_2 in set(range(n_2)) - matched_2:
        total_score += similarity_matrix[:, i_2].max()

    return total_score / n_2