import math

import numpy as np
from scipy.linalg import eigh


def beam(num_elem):
    # Boundary conditions
    restrained_dofs = [1, 0]

    # Length
    l = 1.0 / num_elem
    Cm = 1.0  # rho * A
    Ck = 1.0  # E * I

    # Element mass and stiffness matrices
    m = np.array([[156, 22 * l, 54, -13 * l],
                  [22 * l, 4 * l * l, 13 * l, -3 * l * l],
                  [54, 13 * l, 156, -22 * l],
                  [-13 * l, -3 * l * l, -22 * l, 4 * l * l]]) * Cm * l / 420

    k = np.array([[12, 6 * l, -12, 6 * l],
                  [6 * l, 4 * l * l, -6 * l, 2 * l * l],
                  [-12, -6 * l, 12, -6 * l],
                  [6 * l, 2 * l * l, -6 * l, 4 * l * l]]) * Ck / l ** 3

    # Create global mass and stiffness matrices
    M = np.zeros((2 * num_elem + 2, 2 * num_elem + 2))
    K = np.zeros((2 * num_elem + 2, 2 * num_elem + 2))

    # Assembly of elements
    for i in range(num_elem):
        M_temp = np.zeros((2 * num_elem + 2, 2 * num_elem + 2))
        K_temp = np.zeros((2 * num_elem + 2, 2 * num_elem + 2))
        M_temp[2 * i:2 * i + 4, 2 * i:2 * i + 4] = m
        K_temp[2 * i:2 * i + 4, 2 * i:2 * i + 4] = k
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
    exact_frequency = math.pi ** 2

    # Run example
    for i in range(1, 6):
        M, K, frequencies, evecs = beam(i)
        error = (frequencies[0] - exact_frequency) / exact_frequency * 100.0
        print("Num elements: " + str(i) + ", Fund. Freq: " + str(frequencies[0]) + ", Error: " + str(error) + '%')
