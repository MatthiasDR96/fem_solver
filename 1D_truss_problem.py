import math

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh


def setup():
    
    # Define the coordinate system
    x_axis = np.array([1, 0])
    y_axis = np.array([0, 1])

    # Define the model
    nodes = {1: [0, 0], 2: [0, -0.1], 3: [0, -0.2]}
    degrees_of_freedom = {1: [1], 2: [2], 3: [3]}
    elements = {1: [1, 2], 2: [2, 3]}
    restrained_dofs = [1]
    forces = {1: [0], 2: [0], 3: [100000]}

    # Material properties - AISI 1095, Carbon Steel (Spring Steel)
    densities = {1: 7850, 2: 7850}
    stiffnesses = {1: 210.0e9, 2: 210.0e9}

    # Geometric properties
    areas = {1: math.pi * 0.05 ** 2, 2: math.pi * 0.025 ** 2}

    ndofs = 1 * len(nodes)

    # Assertions
    assert len(densities) == len(elements) == len(stiffnesses) == len(areas)
    assert len(restrained_dofs) < ndofs
    assert len(forces) == len(nodes)

    return {'x_axis': x_axis, 'y_axis': y_axis, 'nodes': nodes, 'degrees_of_freedom': degrees_of_freedom,
            'elements': elements, 'restrained_dofs': restrained_dofs, 'forces': forces, 'densities': densities,
            'stiffnesses': stiffnesses, 'areas': areas, 'ndofs': ndofs}


def plot_nodes(nodes):
    x = [i[0] for i in nodes.values()]
    y = [i[1] for i in nodes.values()]
    size = 400
    offset = size / 4000.
    plt.scatter(x, y, c='y', s=size, zorder=5)
    for i, location in enumerate(zip(x, y)):
        plt.annotate(i + 1, (location[0] - offset, location[1] - offset), zorder=10)


def points(element, properties):
    elements = properties['elements']
    nodes = properties['nodes']
    degrees_of_freedom = properties['degrees_of_freedom']

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

    return from_point, to_point, dofs


def draw_elements(from_point, to_point, element, areas):
    x1 = from_point[0]
    y1 = from_point[1]
    x2 = to_point[0]
    y2 = to_point[1]
    plt.plot([x1, x2], [y1, y2], color='g', linestyle='-', linewidth=7 * areas[element], zorder=1)


def direction_cosine(vec_1, vec_2):
    return np.dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))


def rotation_matrix(element_vector, x_axis, y_axis):
    # Find the direction cosines
    x_proj = direction_cosine(element_vector, x_axis)
    y_proj = direction_cosine(element_vector, y_axis)
    return np.array([[x_proj, y_proj, 0, 0], [0, 0, x_proj, y_proj]])


def get_matrices(properties):
    # Construct the global mass and stiffness matrices
    ndofs = properties['ndofs']
    nodes = properties['nodes']
    elements = properties['elements']
    forces = properties['forces']
    areas = properties['areas']
    x_axis = properties['x_axis']
    y_axis = properties['y_axis']

    plot_nodes(nodes)

    M = np.zeros((ndofs, ndofs))
    K = np.zeros((ndofs, ndofs))

    for element in elements:

        # Find the element geometry
        from_point, to_point, dofs = points(element, properties)
        element_vector = to_point - from_point

        # Display the element
        draw_elements(from_point, to_point, element, areas)

        # Find element mass and stiffness matrices
        length = norm(element_vector)
        rho = properties['densities'][element]
        area = properties['areas'][element]
        E = properties['stiffnesses'][element]

        Cm = rho * area * length / 6
        Ck = E * area / length

        m = np.array([[2, 1], [1, 2]])
        k = np.array([[1, -1], [-1, 1]])

        # Find rotated mass and stiffness matrices
        # tau = rotation_matrix(element_vector, x_axis, y_axis)
        # m_r = tau.T.dot(m).dot(tau)
        # k_r = tau.T.dot(k).dot(tau)

        # Change from element to global coordinates
        index = dofs - 1
        B = np.zeros((len(dofs), ndofs))
        for i in range(len(dofs)):
            B[i, index[i]] = 1.0
        m_rg = B.T.dot(m).dot(B)
        k_rg = B.T.dot(k).dot(B)

        # Add to the global matrices
        M += Cm * m_rg
        K += Ck * k_rg

    # Construct the force vector
    F = []
    for f in forces.values():
        F.extend(f)
    F = np.array(F)

    # Remove the restrained dofs
    K_ = np.array(K)
    remove_indices = np.array(properties['restrained_dofs']) - 1
    for index in remove_indices:
        temp = np.zeros((ndofs,))
        temp[index] = 1
        K_[index, :] = temp
        M[index, :] = temp
        temp = np.zeros((ndofs,)).T
        temp[index] = 1
        K_[:, index] = temp
        M[:, index] = temp
        F[index] = 0

    return M, K, K_, F


def get_stresses(properties, X):
    # Get properties
    x_axis = properties['x_axis']
    y_axis = properties['y_axis']
    elements = properties['elements']
    E = properties['stiffnesses']

    # Find the stresses in each member
    strains = []
    stresses = []
    for element in elements:
        # Find the element geometry
        from_point, to_point, dofs = points(element, properties)
        element_vector = to_point - from_point

        # Find rotation matrix
        # tau = rotation_matrix(element_vector, x_axis, y_axis)
        global_displacement = X[dofs - 1]
        # q = tau.dot(global_displacement)

        # Calculate the strains and stressess
        strain = np.dot((1 / norm(element_vector)) * np.array([-1, 1]), global_displacement)
        strains.append(strain)
        stress = E[element] * strain
        stresses.append(stress)

    return strains, stresses


def show_results(X, strains, stresses, frequencies):
    print("Nodal displacements: " + str(X))
    print("Strains: " + str(strains))
    print("Stressess: " + str(stresses))
    print("Frequencies: " + str(frequencies))
    print("Displacement magnitude: " + str(round(norm(X), 5)))


def main():
    # Problem setup
    properties = setup()

    # Determine the global matrices
    M, K, K_, F = get_matrices(properties)

    # Find the natural frequencies
    evals, evecs = eigh(K_, M)
    frequencies = np.sqrt(evals)

    # Calculate the static displacement of each element
    X = np.linalg.inv(K_).dot(F)

    # Determine the stresses in each element
    strains, stresses = get_stresses(properties, X)

    # Output results
    show_results(X, strains, stresses, frequencies)

    # Total reaction forces
    R = np.dot(K, X) - F

    plt.title('Analysis of Truss Structure')
    plt.show()


if __name__ == "__main__":
    main()
