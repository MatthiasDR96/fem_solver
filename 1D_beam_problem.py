import numpy as np
from numpy.linalg import norm

if __name__ == "__main__":

    # Define the model
    nodes = {1: [0, 0], 2: [1, 0], 3: [2, 0]}
    degrees_of_freedom = {1: [1, 2], 2: [3, 4], 3: [5, 6]}
    elements = {1: [1, 2], 2: [2, 3]}
    restrained_dofs = [1, 2, 3]
    forces = {1: [-6000, -1000], 2: [-6000, 1000], 3: [-40000, 20000]}

    # Material properties - AISI 1095, Carbon Steel (Spring Steel)
    densities = {1: 7850, 2: 7850}
    stiffnesses = {1: 200.0e9, 2: 200.0e9}

    # Geometric properties
    areas = {1: 4.0e-6, 2: 4.0e-6}

    ndofs = 2 * len(nodes)

    K = np.zeros((ndofs, ndofs))
    for element in elements:

        # Find the nodes that the elements connect
        from_node = elements[element][0]
        to_node = elements[element][1]

        # Find the coordinates for each node
        from_point = np.array(nodes[from_node])
        to_point = np.array(nodes[to_node])

        # Find the degrees of freedom for each node
        dofs = list(degrees_of_freedom[from_node])
        dofs.extend(degrees_of_freedom[to_node])
        dofs = np.array(dofs)

        # Calculate length
        l = norm(to_point - from_point)

        # Calculate stiffness matrix
        Ck = stiffnesses[element] * areas[element] / (l ** 3)
        k = np.array([[12, 6 * l, -12, 6 * l],
                      [6 * l, 4 * l * l, -6 * l, 2 * l * l],
                      [-12, -6 * l, 12, -6 * l],
                      [6 * l, 2 * l * l, -6 * l, 4 * l * l]]) * Ck

        # Change from element to global coordinates
        index = dofs - 1
        B = np.zeros((4, ndofs))
        for i in range(4):
            B[i, index[i]] = 1.0
        k_rg = B.T.dot(k).dot(B)

        # Add to global
        K += k_rg

    # Construct the force vector
    F = []
    for f in forces.values():
        F.extend(f)
    F = np.array(F)

    # Remove the restrained dofs
    remove_indices = np.array(restrained_dofs) - 1
    for index in remove_indices:
        temp = np.zeros((ndofs,))
        temp[index] = 1
        K[index, :] = temp
        temp = np.zeros((ndofs,)).T
        temp[index] = 1
        K[:, index] = temp
        F[index] = 0

    # Calculate the static displacement of each element
    X = np.linalg.inv(K).dot(F)

    print(X)

    M_ = []
    S_ = []
    for element in elements:

        # Find the nodes that the elements connect
        from_node = elements[element][0]
        to_node = elements[element][1]

        # Find the coordinates for each node
        from_point = np.array(nodes[from_node])
        to_point = np.array(nodes[to_node])

        # Find the degrees of freedom for each node
        dofs = list(degrees_of_freedom[from_node])
        dofs.extend(degrees_of_freedom[to_node])
        dofs = np.array(dofs) - 1

        # Calculate length
        length = norm(to_point - from_point)

        # Calculate moment
        for e in [-1, 1]:
            M = (stiffnesses[element] * areas[element] / length ** 2) * (6 * e * X[dofs[0]]
                                                                         + (3 * e - 1) * length * X[dofs[1]]
                                                                         - 6 * e * X[dofs[2]] + (3 * e + 1) * length *
                                                                         X[dofs[3]])
            M_.append(M)

        # Calculate shear force
        S = (6 * stiffnesses[element] * areas[element] / length ** 3) * (2 * X[dofs[0]] + length * X[dofs[1]]
                                                                         - 2 * X[dofs[2]] + length * X[dofs[3]])
        S_.append(S)

    print(M_)
    print(S_)
