import numpy as np
from scipy.optimize import linear_sum_assignment


def greedy_matching_similarity(similarity_matrix):
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


def hungarian_matching_similarity(similarity_matrix):
    n_1, n_2 = similarity_matrix.shape  # n_1 <= n_2

    cost_matrix = np.ones((n_2, n_2))
    cost_matrix[:n_1] -= similarity_matrix

    ixs_1, ixs_2 = linear_sum_assignment(cost_matrix)  # ixs_1 = np.arange(n_2)

    if n_1 != n_2:
        ixs_1[n_1:] = np.argmax(similarity_matrix[:, ixs_2[n_1:]], axis=0)

    similarity = similarity_matrix[ixs_1, ixs_2].mean()

    return similarity
