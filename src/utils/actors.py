import numpy as np
import matplotlib.pyplot as plt
import copy

from src.utils.approximators import PolyApproximator
from src.utils.pid import PID


class Actor:
    def __init__(self, radius, max_actuator=1, mass=1, friction_coef=0.1, bounds=10, dim=2):
        self.radius = radius
        self.max_actuator = max_actuator
        self.mass = mass
        self.friction_coef = friction_coef
        self.dim = dim
        self.position: np.ndarray = np.zeros(self.dim, dtype=np.float64)
        self.velocity: np.ndarray = np.zeros(self.dim, dtype=np.float64)
        self.actuator: np.ndarray = np.zeros(self.dim, dtype=np.float64) # force vector
        self.done: bool = False
        self.bounds = bounds

    def reset(self, position, velocity):
        self.position = position.astype(np.float64)
        self.velocity = velocity.astype(np.float64)
        self.actuator = np.zeros(self.dim)
        self.done: bool = False

    # update function only updates the physical state of the actor. the step function is the one that updates the actuator
    def step(self, action):
        self.actuator = np.clip(action, -self.max_actuator, self.max_actuator)

    def update(self, dt):
        acceleration = self.actuator / self.mass
        self.velocity = (1-self.friction_coef) * self.velocity + acceleration * dt
        self.position += self.velocity * dt
        return np.array([self.position, self.velocity]), self.done

    def distance_to(self, actor):
        return np.linalg.norm(self.position - actor.position)

    def kill(self):
        self.done = True
        self.position = np.array([-200, -200], dtype=np.float64)
        self.velocity = np.zeros(self.dim, dtype=np.float64)

    def plot(self, color='b', alpha=0.5):
        plt.scatter(self.position[0], self.position[1], c=color, s=2 * np.pi * self.radius * 100, alpha=alpha)
        # add arrows in the direction of the velocity if velocity is not zero
        if np.linalg.norm(self.velocity) > 1e-2:
            plt.arrow(self.position[0], self.position[1], self.velocity[0], self.velocity[1], width=0.01, color=color)
        if np.linalg.norm(self.actuator) > 1e-2:
            plt.arrow(self.position[0], self.position[1], self.actuator[0], self.actuator[1], width=0.01, color='r')


class Chaser(Actor):
    def __init__(self, pid: PID, radius, max_actuator=1, mass=1, friction_coef=0.1, dim=2):
        super().__init__(radius, max_actuator, mass, friction_coef, dim)
        self.pid = pid
        self.pid.set_output_limits(-self.max_actuator, self.max_actuator)
        self.target_position = np.zeros(self.dim)
        # self.target_velocity = np.zeros(self.dim)
        self.approximator = PolyApproximator(n_samples=6, polyfit_degree=2)
        self.neighbor_repellent_radius = 500 * self.radius
        self.fov_degree = 180
        self.fov_rad = np.deg2rad(self.fov_degree)
        self.error = np.zeros(shape=(2,2))
        self.is_shot_confetti = False
        self.is_ignore_repellent = False
        self.critical_distance = self.neighbor_repellent_radius / 4

    def reset(self, position, velocity, target_position):
        super().reset(position, velocity)
        self.set_target(0, target_position)
        # reset pid by the current position and the target position
        self.pid.reset(self.position, self.target_position)
        self.error[0] = self.target_position - self.position
        self.error[1] = 1.0 * self.error[0]

    def set_target(self, timestamp, target_position):
        self.target_position = target_position
        self.approximator.add_sample(timestamp, *target_position)

    def step(self, dt, action=None, neighbors_position=None):
        # roll self error and put the new error in the second row
        self.error[0] = self.error[1]
        self.error[1] = self.target_position - self.position
        relative_error_velocity = (self.error[1] - self.error[0]) / dt
        distance_to_target = np.linalg.norm(self.error[1])
        self.is_ignore_repellent = distance_to_target < self.critical_distance
        time_to_target = distance_to_target / np.linalg.norm(relative_error_velocity)
        # print("time to target: ", time_to_target)
        time_to_target = np.clip(time_to_target, 0.5 * dt, 500 * dt)
        if self.approximator.is_approximated():
            self.target_position = self.approximator.predict(time_to_target)
        error = self.target_position - self.position + 0*np.random.normal(0, self.radius*1, size=self.dim)
        error_norm = np.linalg.norm(error)
        error_dir = error / error_norm
        self.actuator = error_dir * (min(self.pid.update(error_norm, dt), self.max_actuator)) # this one is good!

        if neighbors_position is not None:
            for neighbor_position in neighbors_position:
                neighbor_error = neighbor_position - self.position
                neighbor_error_norm = np.linalg.norm(neighbor_error)
                if neighbor_error_norm < self.neighbor_repellent_radius:
                    # each chaser has a fov (field of view) in the direction of its velocity
                    # if the neighbor is in the fov, then repell the chaser from the neighbor
                    # if the neighbor is not in the fov, then ignore the neighbor
                    neighbor_error_dir = -neighbor_error / neighbor_error_norm
                    is_in_fov = np.dot(-neighbor_error_dir, self.velocity) / (1e-9 + np.linalg.norm(self.velocity)) > np.cos(self.fov_rad/2)
                    if is_in_fov:
                        repellent_force = neighbor_error_dir * (self.max_actuator / 2) * (1 - neighbor_error_norm / self.neighbor_repellent_radius)
                        if neighbor_error_norm > 0.1 * self.neighbor_repellent_radius:
                            # only parpendicular repellent force is allowed
                            repellent_force_norm = np.linalg.norm(repellent_force)
                            repellent_force_in_velocity_direction = repellent_force * np.dot(repellent_force, self.velocity) / (1e-9 + np.linalg.norm(self.velocity) * np.linalg.norm(repellent_force))
                            repellent_force -= repellent_force_in_velocity_direction
                            repellent_force = repellent_force / (1e-9 + np.linalg.norm(repellent_force)) * repellent_force_norm

                        self.actuator += (1 - self.is_ignore_repellent * 0.5) * repellent_force
        # if norm of the actuator is greater than the max actuator, then normalize the actuator to the max actuator
        if np.linalg.norm(self.actuator) > self.max_actuator:
            self.actuator = self.actuator / np.linalg.norm(self.actuator) * self.max_actuator
        super().step(self.actuator)

    def update(self, dt):
        super().update(dt)
        if self.position[0] > self.bounds or self.position[0] < -self.bounds or self.position[1] > self.bounds or self.position[1] < -self.bounds:
            self.done = True
        return self.velocity, self.done

    def plot(self, color='b', alpha=0.5):
        super().plot(color=color, alpha=alpha)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlim(-self.bounds, self.bounds)
        # plt.ylim(-self.bounds, self.bounds)


class Target(Actor):
    def reset(self, position, velocity=np.zeros(2)):
        super().reset(position, velocity)

    def plot(self, color='g', alpha=0.5):
        super().plot(color=color, alpha=alpha)


class Blast:
    def __init__(self, ax, position, velocity, blast_force=5, radius=0.1, n_particles=25):
        self.n_particles = n_particles
        self.radius = radius
        self.blast_force = blast_force
        self.position = position
        self.velocity = velocity
        self.particles_position = self.position + np.random.normal(0, 0.1, size=(self.n_particles, 2))
        self.particles_velocity = self.velocity + np.random.normal(0, self.blast_force, size=(self.n_particles, 2))
        self.half_life = 1
        self.total_time = 0
        self.done = False
        self.particles_positions_array = []
        for i in range(self.n_particles):
            particle_position, =  ax.plot(0, 0, 'g.', alpha=0.5, markersize=1)
            self.particles_positions_array.append(particle_position)

    def update(self, dt):
        self.total_time += dt
        self.particles_velocity *= 0.99
        self.particles_position += self.particles_velocity * dt
        self.done = self.total_time > self.half_life
        return self.particles_position, self.done

    def plot(self):
        if not self.done:
            for i, particle_position in enumerate(self.particles_positions_array):
                particle_position.set_data(*self.particles_position[i])
        else:
            for particle_position in self.particles_positions_array:
                particle_position.set_data(-200, -200)
