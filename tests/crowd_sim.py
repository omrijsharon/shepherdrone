import numpy as np
import matplotlib.pyplot as plt

from src.utils.helper_functions import \
    calc_distance_vector_matrix, calc_distance_matrix, calc_distance_matrix_from_vector_matrix,\
    calc_orientation, calc_cos_angle_between_orientation_and_distance_vector

def calc_interaction_matrix(position, orientation, radius_matrix, cos_threshold):
    distance_vector_matrix = calc_distance_vector_matrix(position)
    distance_matrix = calc_distance_matrix(position)
    cos_matrix = calc_cos_angle_between_orientation_and_distance_vector(orientation, distance_vector_matrix)
    cos_matrix[cos_matrix < cos_threshold] = 0
    overlap_matrix = distance_matrix - radius_matrix
    overlap_matrix[overlap_matrix > 0] = 0
    interaction_matrix = overlap_matrix * cos_matrix
    return interaction_matrix

if __name__ == '__main__':
    N = 100
    dim = 2
    position = np.random.uniform(-1, 1, size=(dim, N))
    velocity = 0.5*np.random.uniform(-1, 1, size=(dim, N))
    # orientation is a unit vector in the direction of the velocity. if velocity is zero, orientation is the last orientation
    orientation = np.random.uniform(-1, 1, size=(dim, N))
    orientation = calc_orientation(velocity, orientation)
    goal = np.random.uniform(-1, 1, size=(dim, N))
    radius = 0.1
    radius = np.random.uniform(radius, radius, size=N)
    radius_matrix = radius + radius[:, np.newaxis] # this is the radius matrix (N x N). it is symmetric and the diagonal is the sum of the radii
    dt = 8e-3
    bounds = 1
    k = 1e2
    goal_coef = 50
    fov = np.pi*1
    cos_threshold = np.cos(fov/2)
    interaction_matrix = calc_interaction_matrix(position, orientation, radius_matrix, cos_threshold)

    external_force = 0*np.array([1, 0])

    plt.ion()
    for i in range(10000):
        plt.clf()
        interaction_matrix = calc_interaction_matrix(position, orientation, radius_matrix, cos_threshold)
        # add a force that attracts the particles to the center
        f_center = -goal_coef * (position - goal)
        accl = k * (interaction_matrix * calc_distance_vector_matrix(position)).sum(axis=1) + f_center + external_force[:, np.newaxis] + 0.00001*np.random.uniform(-1, 1, size=(dim, N))
        velocity += accl * dt - 0.8 * velocity
        position += velocity * dt
        # wrap position around the bounds
        # position[position > bounds] -= 2 * bounds
        # position[position < -bounds] += 2 * bounds
        orientation = calc_orientation(velocity, orientation)
        if i % 10 == 0:
            # draw particle's goals in pale green
            plt.scatter(goal[0], goal[1], s=radius * 200 * bounds, alpha=0.2, color='g')
            plt.scatter(position[0], position[1], s=radius * 200 * bounds, alpha=0.8)
            # add arrows in the direction of the orientation
            plt.quiver(position[0], position[1], orientation[0], orientation[1], width=0.01, color='b', alpha=0.5, scale=30)
            # draw an arc for the field of view for each particle
            for j in range(N):
                if np.linalg.norm(velocity[:, j]) > 0:
                    # convert orientation unit vector to angle
                    angle = np.arctan2(orientation[1, j], orientation[0, j])
                    plt.plot(position[0, j] + radius[j] * np.cos(np.linspace(angle - fov/2, angle + fov/2, 100)),
                             position[1, j] + radius[j] * np.sin(np.linspace(angle - fov/2, angle + fov/2, 100)), color='r', alpha=0.5)
                    # draw the line from the center of the particle to the edge of the field of view
                    plt.plot([position[0, j], position[0, j] + radius[j] * np.cos(angle + fov/2)], [position[1, j], position[1, j] + radius[j] * np.sin(angle + fov/2)], color='r', alpha=0.5)
                    plt.plot([position[0, j], position[0, j] + radius[j] * np.cos(angle - fov / 2)], [position[1, j], position[1, j] + radius[j] * np.sin(angle - fov / 2)], color='r', alpha=0.5)

            # draw a line between each particle and its goal
            for j in range(N):
                plt.plot([position[0, j], goal[0, j]], [position[1, j], goal[1, j]], color='g', alpha=0.2)
            # square plot
            plt.axis('square')
            plt.xlim(-bounds, bounds)
            plt.ylim(-bounds, bounds)
            plt.pause(0.01)
    plt.show()


