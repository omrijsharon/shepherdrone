import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from time import perf_counter


def calculate_distance_matrix(positions):
    """
    Calculate the distance matrix for a set of positions.

    Parameters:
    positions (numpy.ndarray): An array of positions with shape [N, dim].

    Returns:
    numpy.ndarray: A distance matrix of shape [N, N].
    """
    # Calculate the squared distance matrix
    sq_diff = np.sum((positions[:, np.newaxis] - positions[np.newaxis, :]) ** 2, axis=-1)
    # Take the square root to get the actual distances
    distance_matrix = np.sqrt(sq_diff)

    return distance_matrix


class ExtendedMDS:
    def __init__(self, n_components, altitude_constraint_weight=1.0, anchor_constraint_weight=2.0):
        """
        Extended MDS class constructor.

        :param n_components: Number of dimensions for the embedding.
        :param altitude_constraint_weight: Weight for the altitude constraint in the stress function.
        :param initial_guess: Initial guess for the positions.
        """
        self.n_components = n_components
        self.altitude_constraint_weight = altitude_constraint_weight
        self.anchor_constraint_weight = anchor_constraint_weight
        self.embedding_ = None

    def _stress_function(self, positions_flat, distance_matrix, altitudes, anchor_indices=None, anchor_positions=None):
        """
        Custom stress function with altitude constraints.

        :param positions_flat: Flattened array of positions.
        :param distance_matrix: The matrix of distances between points.
        :param altitudes: Known altitudes of the points.
        :return: The calculated stress value.
        """
        # Reshape the positions array to [N, n_components]
        N = len(altitudes)
        positions = positions_flat.reshape((N, self.n_components))

        # Calculate the reconstructed distances
        reconstructed_distances = np.sqrt(np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=2))

        # Calculate the stress component due to distances
        distance_stress = np.sum((distance_matrix - reconstructed_distances)**2)

        # Calculate the stress component due to altitude constraints
        altitude_stress = np.sum((positions[:, 2] - altitudes)**2)

        # if there are known anchors, add the stress component due to anchor constraints
        if anchor_indices is not None:
            anchor_stress = np.sum((positions[anchor_indices] - anchor_positions)**2)
            # Total stress is the sum of both components
            return distance_stress + self.altitude_constraint_weight * altitude_stress + self.anchor_constraint_weight * anchor_stress

        # Total stress is the sum of both components
        return distance_stress + self.altitude_constraint_weight * altitude_stress



    def fit(self, distance_matrix, altitudes, initial_guess=None, anchor_indices=None, anchor_positions=None):
        """
        Fit the Extended MDS model to the distance matrix with altitude constraints.

        :param distance_matrix: The matrix of distances between points.
        :param altitudes: Known altitudes of the points.
        """
        N = distance_matrix.shape[0]

        # Use the initial guess if provided, otherwise start with random positions
        if not (initial_guess is not None and initial_guess.shape == (N, self.n_components)):
            initial_guess = np.random.rand(N, self.n_components)

        # Flatten the initial positions for the optimizer
        initial_positions_flat = initial_guess.flatten()

        t0 = perf_counter()
        # Optimize the positions
        result = minimize(self._stress_function, initial_positions_flat,
                          args=(distance_matrix, altitudes, anchor_indices, anchor_positions), method='L-BFGS-B', tol=1e-6)
        t1 = perf_counter()
        # print the time it took to optimize in ms and in Hz
        print(f'Optimization took {(t1 - t0) * 1000:.3f} ms')
        print(f'Optimization took {1 / (t1 - t0):.3f} Hz')
        # Reshape the result and store in embedding_
        self.embedding_ = result.x.reshape((N, self.n_components))

        return self

    def get_embedding(self):
        """
        Return the embedding.
        """
        return self.embedding_

if __name__ == '__main__':
    N = 7
    dim = 3
    positions = np.random.uniform(-10, 10, size=(N, dim))
    positions[:, 2] += 10
    distance_matrix = calculate_distance_matrix(positions)
    altitudes = positions[:, 2]
    emds = ExtendedMDS(n_components=3, altitude_constraint_weight=1.0)
    initial_guess = np.random.uniform(-10, 10, size=(N, dim))
    initial_guess[:, 2] += 10
    initial_guess[:3, :] = positions[:3, :]
    initial_guess = positions + np.random.randn(N, dim) * 2
    # emds.fit(distance_matrix, altitudes, initial_guess=initial_guess, anchor_indices=[0, 1, 2], anchor_positions=positions[:3, :])
    emds.fit(distance_matrix, altitudes, initial_guess=initial_guess)
    estimated_positions = emds.get_embedding()
    # plot the estimated positions on top of the true positions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
    ax.scatter(estimated_positions[:, 0], estimated_positions[:, 1], estimated_positions[:, 2], c='b', marker='o')
    # plot title the score which is the MSE between the true distance matrix and the reconstructed distance matrix
    score = np.mean((distance_matrix - calculate_distance_matrix(estimated_positions))**2)
    ax.set_title(f'Score: {score:.9f}')
    plt.show()

