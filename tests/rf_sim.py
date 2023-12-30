import numpy as np

c = 3e8
bandwidth = 40e6 # Hz
power_level = 100 # mW
f = 2400e6 # Hz
lm = c / f
alpha = 2

def pathloss_dB(d, f):
    return 20 * np.log10(d) + 20 * np.log10(f) - 147.55

def fading():
    return np.random.randn() + 1j * np.random.randn()

def h_freespace(d, f):
    return 10 ** (-pathloss_dB(d, f) / 20)

if __name__ == '__main__':
    pass