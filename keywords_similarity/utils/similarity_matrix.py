import numpy as np


def calculate_similarity_matrix(group_1, group_2, similarity_function):
    n_1, n_2 = len(group_1), len(group_2)

    similarity_matrix = np.zeros((n_1, n_2))

    for i in range(n_1):
        for j in range(n_2):
            s_i, s_j = group_1[i], group_2[j]
            similarity_matrix[i, j] = similarity_function(s_i, s_j)

    return similarity_matrix
