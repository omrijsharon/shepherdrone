import numpy as np
import matplotlib.pyplot as plt

class Herd:
    def __init__(self, n_particles, dim, radius_vector, dt=1e-1, k=1e1, position=None, velocity=None):
        self.n_particles = n_particles
        self.dim = dim
        self.dt = dt
        self.k = k
        self.bounds = 10
        self.radius = radius_vector
        if position is None:
            self.position = np.random.uniform(-self.bounds, self.bounds, size=(n_particles, dim))
        else:
            self.position = position
        if velocity is None:
            self.velocity = np.random.uniform(-1, 1, size=(n_particles, dim))
        else:
            self.velocity = velocity

        self.distance_vector_matrix = np.zeros((n_particles, n_particles, dim))
        self.distance_matrix = np.zeros((n_particles, n_particles))
        self.radius_matrix = self.calculate_radius_matrix()
        self.overlap_matrix = np.zeros((n_particles, n_particles))

    def calculate_distance_vector_matrix(self):
        """Calculate the distance vector matrix (N x N x dim)"""
        for d in range(self.dim):
            self.distance_vector_matrix[:, :, d] = self.position[:, d] - self.position[:, d][:, np.newaxis]

    def calculate_distance_matrix(self):
        """Calculate the distance matrix (N x N)"""
        self.distance_matrix = np.linalg.norm(self.distance_vector_matrix, axis=2)

    def calculate_radius_matrix(self):
        """Calculate the radius matrix (N x N)"""
        return self.radius + self.radius[:, np.newaxis]

    def calculate_overlap_matrix(self):
        """Check if there is overlap between the circles, if there is, returns the distance vector, else returns -1"""
        self.overlap_matrix = self.distance_matrix - self.radius_matrix
        self.overlap_matrix[self.overlap_matrix > 0] = 0
        self.overlap_matrix[self.overlap_matrix < 0] = 1
        self.overlap_matrix -= np.eye(self.overlap_matrix.shape[0])

    def calculate_mask(self):
        self.calculate_distance_vector_matrix()
        self.calculate_distance_matrix()
        self.calculate_overlap_matrix()
        return self.overlap_matrix[:, :, np.newaxis]

    def update(self):
        accl = - self.k * (self.calculate_mask() * self.distance_vector_matrix).sum(axis=1)
        self.velocity += accl * self.dt
        self.position += self.velocity * self.dt

    def plot(self):
        plt.clf()
        plt.scatter(self.position[:, 0], self.position[:, 1], s=self.radius * 100 * self.bounds, alpha=0.5)
        plt.xlim(-self.bounds, self.bounds)
        plt.ylim(-self.bounds, self.bounds)
        plt.pause(0.0000001)


if __name__ == '__main__':
    dt = 1e-2
    k = 1000
    n_particles = 20
    radius_vector = np.random.uniform(0.1, 2, size=n_particles)
    herd = Herd(n_particles=n_particles, dim=2, dt=dt, k=k, radius_vector=radius_vector)
    for _ in range(1000):
        herd.update()
        herd.plot()
    plt.show()