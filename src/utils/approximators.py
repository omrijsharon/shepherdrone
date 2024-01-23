import numpy as np

# the following class job is to get data of the target (time, x, y) and make a rough estimation (using a second order polyfit) of the target's position in the next couple of seconds
class PolyApproximator:
    def __init__(self, n_samples, polyfit_degree=2):
        self.n_samples = n_samples
        self.polyfit_degree = polyfit_degree
        self.time = np.zeros(n_samples)
        self.x = np.zeros(n_samples)
        self.y = np.zeros(n_samples)
        self.x_polyfit = np.zeros(polyfit_degree + 1)
        self.y_polyfit = np.zeros(polyfit_degree + 1)
        self.counter = 0

    def add_sample(self, timestamp, x, y):
        self.time = np.roll(self.time, -1)
        self.x = np.roll(self.x, -1)
        self.y = np.roll(self.y, -1)
        self.time[-1] = timestamp
        self.x[-1] = x
        self.y[-1] = y
        self.counter += 1

    def calc_polyfit(self):
        self.x_polyfit = np.polyfit(self.time - self.time[0], self.x, self.polyfit_degree)
        self.y_polyfit = np.polyfit(self.time - self.time[0], self.y, self.polyfit_degree)

    def predict(self, dt):
        self.calc_polyfit()
        timestamp = self.time[-1] + dt - self.time[0]
        return np.array([np.polyval(self.x_polyfit, timestamp), np.polyval(self.y_polyfit, timestamp)])

    def is_approximated(self):
        return self.counter >= self.n_samples

    def __len__(self):
        return self.counter