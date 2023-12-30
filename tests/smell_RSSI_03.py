import numpy as np
import matplotlib.pyplot as plt
from time import sleep

class Target:
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='g', s=self.radius * 1000, alpha=0.5)

class Chaser:
    def __init__(self, radius, dim=2, dt=1e-2):
        self.dt = dt
        self.mass = .5
        self.dim = dim
        self.friction = 0.2
        self.position: np.ndarray = np.zeros(self.dim)
        self.velocity: np.ndarray = np.zeros(self.dim)
        assert radius > 0 and isinstance(radius, (int, float)), "Radius must be a positive number"
        self.radius = float(radius)
        self.prev_rssi: float = -100
        self.done: bool = False
        self.bounds = 100
        self.rssi_filter = 0.4

    def reset(self, position, target: Target):
        self.position = position
        self.velocity = np.zeros(self.dim)
        self.prev_rssi = self.distance2rssi(self.distance_to(target), noise=1e-6)
        self.done: bool = False
        return self.velocity, self.prev_rssi

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='b', s=self.radius * 100, alpha=0.5)
        # add arrows in the direction of the velocity
        plt.arrow(self.position[0], self.position[1], self.velocity[0], self.velocity[1], width=0.01, color='b')
        plt.xlim(-self.bounds, self.bounds)
        plt.ylim(-self.bounds, self.bounds)

    def step(self, force_vector, target: Target):
        raw_rssi = self.distance2rssi(self.distance_to(target), noise=1e-3)
        self.prev_rssi = self.rssi_filter * raw_rssi + (1-self.rssi_filter) * self.prev_rssi
        acceleration = force_vector / self.mass
        self.velocity = (1-self.friction) * self.velocity + acceleration * self.dt
        self.position += self.velocity * self.dt
        if self.distance_to(target) < (self.radius + target.radius):
            self.done = True
        if self.position[0] > self.bounds or self.position[0] < -self.bounds or self.position[1] > self.bounds or self.position[1] < -self.bounds:
            self.done = True
        return self.velocity, self.prev_rssi, self.done, {}

    def distance_to(self, target: Target):
        return np.linalg.norm(self.position - target.position)

    def distance2rssi(self, distance, noise=0.0):
        return -10 * np.log10(distance) + np.random.normal(0, noise, size=(1,))

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def normalize(vector):
    return vector / np.linalg.norm(vector)


def pol2cart(theta, rho):
    return rho * np.array([np.cos(theta), np.sin(theta)])


if __name__ == '__main__':
    dim = 3
    dt = 1e-1
    sub_step_size = 25
    initial_altitude = 15
    env = Chaser(radius=0.01, dim=dim, dt=dt)
    target_random_position = 0.1 * np.random.uniform(-1, 1, size=dim)
    target_random_position[2] = 0
    target = Target(target_random_position, radius=0.01)
    chaser_random_position = np.random.uniform(-env.bounds, env.bounds, size=dim)
    chaser_random_position[2] = initial_altitude
    angle_action = 45
    min_angle = 5
    max_angle = 60
    counter_threshold = 6
    # first step
    vel, prev_rssi = env.reset(chaser_random_position, target)
    force_vector_magnitude = 0.75
    angle = np.random.uniform(-180, 180)
    force_vector = pol2cart(np.deg2rad(angle), force_vector_magnitude)
    force_vector = np.append(force_vector, 0)

    for i in range(sub_step_size):
        vel, rssi, done, info = env.step(force_vector, target)
    diff_rssi = rssi - prev_rssi
    prev_rssi = rssi.copy()
    angle += angle_action
    force_vector = pol2cart(np.deg2rad(angle), force_vector_magnitude)
    force_vector = np.append(force_vector, 0)

    for i in range(sub_step_size):
        vel, rssi, done, info = env.step(force_vector, target)
    prev_diff_rssi = diff_rssi.copy()
    diff_rssi = rssi - prev_rssi
    prev_rssi = rssi.copy()
    if diff_rssi > prev_diff_rssi:
        angle_action *= -1
    angle += angle_action
    force_vector = pol2cart(np.deg2rad(angle), force_vector_magnitude)
    force_vector = np.append(force_vector, 0)
    chaser_position = np.array([env.position]).reshape(1, dim)
    step = 0
    counter = 0
    while not env.done:
        plt.clf()
        rssi_array = np.zeros(sub_step_size)
        for i in range(sub_step_size):
            vel, rssi, done, info = env.step(force_vector, target)
            rssi_array[i] = rssi
        prev_diff_rssi = diff_rssi.copy()
        diff_rssi = rssi - prev_rssi
        prev_rssi = rssi.copy()
        rssi_acc = diff_rssi - prev_diff_rssi
        if rssi_acc < 0:
            angle_action *= -0.8
            counter = 0
        else:
            counter += 1
        if (counter > counter_threshold) | ((prev_diff_rssi < 0) & (diff_rssi < 0)):
            angle_action *= 2
            counter = 0
        if abs(angle_action) < min_angle:
            angle_action = np.sign(angle_action) * min_angle
        elif abs(angle_action) > max_angle:
            angle_action = np.sign(angle_action) * max_angle
        print(angle_action, diff_rssi, diff_rssi)
        angle += angle_action
        force_vector = pol2cart(np.deg2rad(angle), force_vector_magnitude)
        force_vector = np.append(force_vector, 0)

        chaser_position = np.concatenate((chaser_position, np.array([env.position]).reshape(1, dim)), axis=0)
        if step % 1 == 0:
            plt.subplot(1, 2, 1)
            plt.arrow(env.position[0], env.position[1], force_vector[0], force_vector[1], width=0.01, color='r')
            # plot chaser position
            plt.plot(chaser_position[:, 0], chaser_position[:, 1], c='b', alpha=0.5)
            env.plot()
            target.plot()
            # make plot with equal aspect ratio and equal axes
            plt.axis('square')
            plt.xlim(-env.bounds, env.bounds)
            plt.ylim(-env.bounds, env.bounds)
            # plt.title(f"RSSI: {env.prev_rssi:.4f}")
            plt.subplot(1, 2, 2)
            plt.plot(rssi_array)
            # plt.subplot(1, 3, 3)
            # plt.plot(np.diff(np.array(diff_rssi)))
            plt.pause(1e-4)
        step += 1