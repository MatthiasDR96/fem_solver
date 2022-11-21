import numpy as np


if __name__ == "__main__":

    # Define the model
    nodes = {1: [0, 0], 2: [0.030, 0.010], 3: [0.020, 0.030]}
    degrees_of_freedom = {1: [1, 2], 2: [3, 4], 3: [5, 6]}
    elements = {1: [1, 2, 3]}
    restrained_dofs = [1, 2, 4]
    forces = {1: [-6000, -1000], 2: [-6000, 1000], 3: [-40000, 20000]}

    # Material properties - AISI 1095, Carbon Steel (Spring Steel)
    densities = {1: 7850, 2: 7850}
    stiffnesses = {1: 200.0e9, 2: 200.0e9}

    for element in elements:
        U = np.array([0, 0, 2, 0, 1, -1]).T*1.0e-6
        i_pos = nodes[elements[element][0]]
        j_pos = nodes[elements[element][1]]
        k_pos = nodes[elements[element][2]]
        det_J = (i_pos[0] - k_pos[0]) * (i_pos[1] - k_pos[1]) - (j_pos[0] - k_pos[0]) * (j_pos[1] - k_pos[1])
        B = (1/det_J) * np.array([[(j_pos[1] - k_pos[1]), 0, (k_pos[1] - i_pos[1]), 0, (i_pos[1] - j_pos[1]), 0],
                                 [0, (k_pos[0] - j_pos[0]), 0, (i_pos[0] - k_pos[0]), 0, (j_pos[0] - i_pos[0])],
                                 [(k_pos[0] - j_pos[0]), (j_pos[1] - k_pos[1]), (i_pos[0] - k_pos[0]),
                                  (k_pos[1] - i_pos[1]), (j_pos[0] - i_pos[0]), (i_pos[1] - j_pos[1])]])

        strain = np.dot(B, U)
        print(strain)

