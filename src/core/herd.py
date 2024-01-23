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
        self.velociradius = radius_vector
        if position is None:
            self.position = np.random.uniform(-self.bounds/2, self.bounds/2, size=(n_particles, dim))
        else:
            self.position = position
        if velocity is None:
            self.velocity = 0.00001 * np.random.uniform(-1, 1, size=(n_particles, dim))
        else:
            self.velocity = velocity

        self.distance_vector_matrix = np.zeros((n_particles, n_particles, dim))
        self.distance_matrix = np.zeros((n_particles, n_particles))
        self.radius_matrix = self.calculate_radius_matrix()
        self.overlap_matrix = np.zeros((n_particles, n_particles))
        self.velocidist_vector_matrix = np.zeros((n_particles, n_particles, dim))
        self.velocidist_matrix = np.zeros((n_particles, n_particles))
        self.velociradius_matrix = self.calculate_velociradius_matrix()
        self.velocioverlap_matrix = np.zeros((n_particles, n_particles))

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

    def calculate_velociradius_matrix(self):
        """Calculate the radius matrix (N x N)"""
        return self.velociradius + self.velociradius[:, np.newaxis]

    def calculate_overlap_matrix(self):
        """Check if there is overlap between the circles, if there is, returns the distance vector, else returns -1"""
        self.overlap_matrix = self.distance_matrix - self.radius_matrix
        self.overlap_matrix[self.overlap_matrix > 0] = 0
        self.overlap_matrix[self.overlap_matrix < 0] = 1
        self.overlap_matrix -= np.eye(self.overlap_matrix.shape[0])

    def calculate_velocioverlap_matrix(self):
        """Check if there is overlap between the circles, if there is, returns the distance vector, else returns -1"""
        self.velocioverlap_matrix = self.velocidist_matrix - self.velociradius_matrix
        self.velocioverlap_matrix[self.velocioverlap_matrix > 0] = 0
        self.velocioverlap_matrix[self.velocioverlap_matrix < 0] = 1
        self.velocioverlap_matrix -= np.eye(self.velocioverlap_matrix.shape[0])

    def calculate_mask(self):
        self.calculate_distance_vector_matrix()
        self.calculate_distance_matrix()
        self.calculate_overlap_matrix()
        return self.overlap_matrix[:, :, np.newaxis]

    def calculate_velocimask(self):
        self.calculate_velocidist_vector_matrix()
        self.calculate_velocidist_matrix()
        self.calculate_velocioverlap_matrix()
        return self.velocioverlap_matrix[:, :, np.newaxis]

    def calculate_velocidist_vector_matrix(self):
        """Calculate the distance vector matrix (N x N x dim)"""
        # unit_velocity = self.velocity / np.linalg.norm(self.velocity, axis=1)[:, np.newaxis]
        unit_velocity = self.velocity
        for d in range(self.dim):
            self.velocidist_vector_matrix[:, :, d] = unit_velocity[:, d] - unit_velocity[:, d][:, np.newaxis]

    def calculate_velocidist_matrix(self):
        """Calculate the distance matrix (N x N)"""
        self.velocidist_matrix = np.linalg.norm(self.velocidist_vector_matrix, axis=2)

    def update(self):
        mask = self.calculate_mask()
        accl = - self.k * (mask * self.distance_vector_matrix).sum(axis=1)
        # add a force that aligns the particles'
        self.calculate_velocidist_vector_matrix()
        velocimask = self.calculate_velocimask()
        velocity_force = (velocimask * self.velocidist_vector_matrix).sum(axis=1)
        # normalize velocity_force to unit vector where the norm is greater than 1
        norm_velocity_force = np.linalg.norm(velocity_force, axis=1)
        velocity_force[norm_velocity_force > 1] = velocity_force[norm_velocity_force > 1] / norm_velocity_force[norm_velocity_force > 1][:, np.newaxis]
        accl += - 1e3 * velocity_force
        print(accl)
        self.velocity += accl * self.dt
        # normalize velocity to unit vector where the norm is greater than 1
        norm_velocity = np.linalg.norm(self.velocity, axis=1)
        self.velocity[norm_velocity > 1] = self.velocity[norm_velocity > 1] / norm_velocity[norm_velocity > 1][:, np.newaxis]
        self.position += self.velocity * self.dt

    def plot(self):
        plt.clf()
        plt.scatter(self.position[:, 0], self.position[:, 1], s=self.radius * 100 * self.bounds, alpha=0.5)
        # plot arrows of the unit velocities of the particles
        unit_velocity = self.velocity / np.linalg.norm(self.velocity, axis=1)[:, np.newaxis]
        plt.quiver(self.position[:, 0], self.position[:, 1], unit_velocity[:, 0], unit_velocity[:, 1], width=0.01, color='b', alpha=0.5, scale=30)
        plt.xlim(-self.bounds, self.bounds)
        plt.ylim(-self.bounds, self.bounds)
        plt.pause(0.0000001)


if __name__ == '__main__':
    dt = 5e-2
    k = 0e1
    n_particles = 20
    radius_vector = 1+ np.random.uniform(0.3, 1.5, size=n_particles)
    herd = Herd(n_particles=n_particles, dim=2, dt=dt, k=k, radius_vector=radius_vector)
    for _ in range(1000):
        herd.update()
        herd.plot()
    plt.show()