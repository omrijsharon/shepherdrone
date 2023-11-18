import numpy as np
import matplotlib.pyplot as plt

def calc_distance_vector_matrix(coordinates):
    """Calculate the distance vector matrix (N x N x dim)"""
    return coordinates[:, :, None] - coordinates[:, None, :]

def calc_distance_matrix(coordinates):
    """Calculate the distance matrix (N x N)"""
    return np.linalg.norm(calc_distance_vector_matrix(coordinates), axis=0)

def calc_distance_matrix_from_vector_matrix(distance_vector_matrix):
    """Calculate the distance matrix (N x N)"""
    return np.linalg.norm(distance_vector_matrix, axis=0)

def calc_orientation(velocity, last_orientation):
    """Calculate the orientation of the velocity [dim x N]"""
    orientation = velocity / np.linalg.norm(velocity, axis=0)
    orientation[:, np.where(np.linalg.norm(velocity, axis=0) == 0)] = last_orientation[:, np.where(np.linalg.norm(velocity, axis=0) == 0)]
    return orientation

def calc_cos_angle_between_orientation_and_distance_vector(orientation, distance_vector_matrix):
    """Calculate the cos angle between the orientation and the distance vector matrix [N x N]"""
    distance_vector_matrix = distance_vector_matrix + np.eye(distance_vector_matrix.shape[1])[None, :, :]
    return np.sum(orientation[:, :, None] * distance_vector_matrix, axis=0) / np.linalg.norm(distance_vector_matrix, axis=0)

class SquareCell:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.n_coordinates = coordinates.shape[1]
        self.mean_coordinates = np.mean(coordinates, axis=1)
        self.center = self.mean_coordinates
        # self.radius = np.max(np.abs(coordinates - self.mean_coordinates[:, None]), axis=1)
        self.n_dim = coordinates.shape[0]
        self.n_squares = 2**self.n_dim
        self.squares = None
        self.is_leaf = True
        self.children = None
        self.parent = None
        self.is_root = False
        self.is_full = False
        self.is_divided = False
        self.color = np.random.uniform(0, 1, size=self.n_dim)
        if self.n_coordinates > 3:
            self.divide()
        else:
            self.is_full = True

    def divide(self):
        self.is_divided = True
        self.is_leaf = False
        self.children = []
        # divide coordinates into 2**n_dim squares by the mean of the coordinates
        for i in range(self.n_squares):
            # create a mask for each square
            mask = np.zeros(self.n_coordinates, dtype=bool)
            for j in range(self.n_dim):
                mask |= (self.coordinates[j] > self.mean_coordinates[j]) ^ ((i >> j) & 1)
            # create a new square with the masked coordinates
            new_square = SquareCell(self.coordinates[:, mask])
            new_square.parent = self
            self.children.append(new_square)

    def plot(self):
        if self.is_leaf:
            # color must be with the same number as the number of coordinates
            color = np.random.uniform(0, 1, size=self.n_dim)
            plt.scatter(self.coordinates[0], self.coordinates[1], s=10, c=color)
        else:
            for child in self.children:
                child.plot()



if __name__ == '__main__':
    N = 100
    dim = 2
    coordinates = np.random.uniform(-1, 1, size=(dim, N))
    plt.scatter(coordinates[0], coordinates[1], s=10)
    grid = SquareCell(coordinates)
    grid.plot()
    plt.show()