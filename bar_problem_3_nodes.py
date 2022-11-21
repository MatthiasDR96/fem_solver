import math

import numpy as np
from scipy.linalg import eigh


def bar(num_elem):
    # Boundary conditions
    restrained_dofs = [0, ]

    # Create mass and stiffness matrices for a bar
    m = np.array([[4, 2, -1], [2, 16, 2], [-1, 2, 4]]) / (30.0 * num_elem)
    k = np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]]) / 3.0 * num_elem

    # Create global mass and stiffness matrices
    M = np.zeros((2 * num_elem + 1, 2 * num_elem + 1))
    K = np.zeros((2 * num_elem + 1, 2 * num_elem + 1))

    # Assembly of elements
    for i in range(num_elem):
        M_temp = np.zeros((2 * num_elem + 1, 2 * num_elem + 1))
        K_temp = np.zeros((2 * num_elem + 1, 2 * num_elem + 1))
        first_node = 2 * i
        M_temp[first_node:first_node + 3, first_node:first_node + 3] = m
        K_temp[first_node:first_node + 3, first_node:first_node + 3] = k
        M += M_temp
        K += K_temp

    # Remove fixed degrees of freedom
    for dof in restrained_dofs:
        for i in [0, 1]:
            M = np.delete(M, dof, axis=i)
            K = np.delete(K, dof, axis=i)

    # Eigenvalue problem
    evals, evecs = eigh(K, M)
    frequencies = np.sqrt(evals)
    return M, K, frequencies, evecs


if __name__ == '__main__':

    # Exact freq
    exact_frequency = math.pi / 2

    # Run example
    for i in range(1, 11):
        M, K, frequencies, evecs = bar(i)
        error = (frequencies[0] - exact_frequency) / exact_frequency * 100.0
        print("Num elements: " + str(i) + ", Fund. Freq: " + str(frequencies[0]) + ", Error: " + str(error) + '%')
