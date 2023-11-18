import numpy as np
import matplotlib.pyplot as plt

class Target:
    def __init__(self, position, radius, dt=1e-1):
        self.dt = dt
        self.position = position
        self.velocity = np.random.uniform(-1, 1, size=2)
        self.velocity /= np.linalg.norm(self.velocity)
        self.velocity *= 0.1
        self.radius = radius
        self.bounds = 3
    def update(self):
        self.position += self.velocity * self.dt

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='g', s=2 * np.pi * self.radius * 100 * self.bounds, alpha=0.5)

class Chaser:
    def __init__(self, radius, dim=2, dt=1e-1):
        self.dt = dt
        self.mass = 1
        self.dim = dim
        self.friction = 0.1
        self.position: np.ndarray = np.zeros(self.dim)
        self.velocity: np.ndarray = np.zeros(self.dim)
        assert radius > 0 and isinstance(radius, (int, float)), "Radius must be a positive number"
        self.radius = float(radius)
        self.prev_rssi: float = -100
        self.done: bool = False
        self.bounds = 3
        self.rssi_filter = 0.2

    def reset(self, position, target: Target):
        self.position = position
        self.velocity = np.zeros(2)
        self.prev_rssi = self.distance2rssi(self.distance_to(target, noise=1e-2))
        self.done: bool = False

    def plot(self):
        plt.scatter(self.position[0], self.position[1], c='b', s=2 * np.pi * self.radius * 100 * self.bounds, alpha=0.5)
        # add arrows in the direction of the velocity
        plt.arrow(self.position[0], self.position[1], self.velocity[0], self.velocity[1], width=0.01, color='b')

    def step(self, force_vector, target: Target):
        rssi = self.rssi_filter * self.distance2rssi(self.distance_to(target, noise=1e-2)) + (1-self.rssi_filter) * self.prev_rssi
        acceleration = force_vector / self.mass
        self.velocity = (1-self.friction) * self.velocity + acceleration * self.dt
        self.position += self.velocity * self.dt
        if self.distance_to(target) < (self.radius + target.radius):
            self.done = True
        # if self.position[0] > self.bounds or self.position[0] < -self.bounds or self.position[1] > self.bounds or self.position[1] < -self.bounds:
        #     self.done = True
        diff_rssi = rssi - self.prev_rssi
        self.prev_rssi = rssi
        return self.velocity, diff_rssi, self.done, {}

    def distance_to(self, target: Target, noise=0.0):
        return np.linalg.norm(self.position - target.position + np.random.normal(0, noise, size=self.dim))

    def distance2rssi(self, distance):
        return -10 * np.log10(distance)

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

def normalize(vector):
    return vector / np.linalg.norm(vector)

if __name__ == '__main__':
    dim = 2
    dt = 1e-1
    env = Chaser(radius=0.01, dim=dim, dt=dt)
    target_random_position = 0.1 * np.random.uniform(-1, 1, size=dim)
    target = Target(target_random_position, radius=0.01, dt=dt)
    chaser_random_position = np.random.uniform(-env.bounds, env.bounds, size=dim)
    env.reset(chaser_random_position, target)
    force_vector = 0.9 * normalize(np.random.uniform(-1, 1, size=dim))
    force_vector = np.clip(np.linalg.norm(force_vector), 0.1, 0.5) * normalize(force_vector)
    diff_rssi = []
    for i in range(2):
        obs, reward, done, info = env.step(force_vector, target)
        diff_rssi.append(reward)
    chaser_position = np.array([env.position]).reshape(1, dim)
    target_position = np.array([target.position]).reshape(1, dim)
    step = 0
    direction = 1
    rapid_change_counter = 0
    is_changed = False
    while not env.done:
        plt.clf()
        target.update()
        obs, reward, done, info = env.step(force_vector=force_vector, target=target)
        chaser_position = np.concatenate((chaser_position, np.array([env.position]).reshape(1, dim)), axis=0)
        target_position = np.concatenate((target_position, np.array([target.position]).reshape(1, dim)), axis=0)
        diff_rssi.append(reward)
        velocity = obs
        if step % 4 == 0:
            if reward < 0: # getting farther
                force_vector = np.dot(rotation_matrix(direction * np.pi / 2), normalize(velocity)) * np.linalg.norm(force_vector)
                rapid_change_counter += 1
                # if rapid_change_counter > 1:
                #     force_vector *= 0.75
                if rapid_change_counter % 12 == 11:
                    direction *= -1
                print("getting farther away")
            else:
                rapid_change_counter = 0
                if np.diff(np.diff(np.array(diff_rssi)))[-1] < 0: # getting closer but slower
                    # change the force vector by plus or minus 45 degrees, randomly
                    force_vector = np.dot(rotation_matrix(direction * np.pi/3), normalize(velocity)) * np.linalg.norm(force_vector)
                    print("getting closer but slower")
                    # is_changed = False
                    # force_vector /= 0.75
                else: # getting closer and faster
                    if not is_changed:
                        direction *= -1
                        is_changed = True
                    # change the force vector by -45 degrees
                    # force_vector = np.dot(rotation_matrix(-np.pi/12), normalize(velocity)) * np.linalg.norm(force_vector)
                    print("getting closer and faster")
            plt.subplot(1, 3, 1)
            plt.plot(chaser_position[:, 0], chaser_position[:, 1], c='b', alpha=0.5)
            plt.plot(target_position[:, 0], target_position[:, 1], c='g', alpha=0.5)
            plt.arrow(env.position[0], env.position[1], force_vector[0], force_vector[1], width=0.01, color='r')
            env.plot()
            target.plot()
            # xlim and ylim are centered around the target position with a margin of env.bounds
            plt.xlim(target.position[0] - env.bounds, target.position[0] + env.bounds)
            plt.ylim(target.position[1] - env.bounds, target.position[1] + env.bounds)
            plt.title(f"RSSI: {env.prev_rssi:.4f}")
            # make the plot square
            plt.gca().set_aspect('equal', adjustable='box')
            plt.subplot(1, 3, 2)
            plt.plot(np.array(diff_rssi))
            plt.subplot(1, 3, 3)
            plt.plot(np.diff(np.array(diff_rssi)))
            plt.pause(1e-6)
        diff_rssi.pop(0)
        step += 1
        # clip force vector norm
        force_vector = np.clip(np.linalg.norm(force_vector), 0.9, 1.0) * normalize(force_vector)
