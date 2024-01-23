import matplotlib.pyplot as plt
import numpy as np
from src.utils.actors import Chaser, Target, Blast
from src.utils.pid import PID


class CursorTracker:
    def __init__(self, ax):
        self.ax = ax
        self.setpoint = (0, 0)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return
        self.setpoint = (event.xdata, event.ydata)


def update_particle_position(current_position, setpoint, ema_coeff=1):
    # Here, implement your PID control logic to update the position
    # For simplicity, I'm just moving the particle halfway to the setpoint
    new_position = (current_position[0] + ema_coeff * (setpoint[0] - current_position[0]),
                    current_position[1] + ema_coeff * (setpoint[1] - current_position[1]))
    return new_position


if __name__ == '__main__':
    dt = 1/100 # 1/Hz
    dt_action = 1/20 # 1/Hz
    control_to_physics_ratio = np.round(dt_action / dt).astype(int)

    n_chasers = 12
    bounds = 100
    pid = PID(65,0, 0)
    chasers = [Chaser(pid=pid, radius=0.05, max_actuator=80, friction_coef=0.04, dim=2) for _ in range(n_chasers)]
    target = Target(radius=2.0)

    # target position is random in the top right and left quadrants
    target.reset(np.array([np.random.uniform(-9, 9), np.random.uniform(0, 9)]))
    # reset chaser positions to random positions between -100 and -90 in both axes
    [chaser.reset(np.random.uniform(-100, -90, size=2), np.zeros(2), target.position) for chaser in chasers]

    fig, ax = plt.subplots()
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    target_position, = ax.plot(0, 0, 'ro', markersize=5)
    chaser_positions = []
    for i in range(n_chasers):
        chaser_position, = ax.plot(0, 0, 'bo', markersize=2)
        chaser_positions.append(chaser_position)

    # Set up cursor tracker
    tracker = CursorTracker(ax)
    blasts = []
    pixel_noise_level = target.radius * 0
    # while the distance between the chaser and the target is greater than the radius of the target + the radius of the chaser
    t = 0
    done = False
    # while chaser.distance_to(target) > (chaser.radius + target.radius):
    while not done:
        # plt.clf()

        target.position = np.array(update_particle_position(target.position, tracker.setpoint, ema_coeff=0.008))
        target_position.set_data(*target.position)

        if (int(t/dt) % control_to_physics_ratio) == 0:
            [chaser.set_target(t, target.position) for chaser in chasers]
            # prepare a chaser_position_array
            chaser_position_array = np.zeros((n_chasers, 2))
            for i, chaser in enumerate(chasers):
                chaser_position_array[i] = chaser.position
            for i, chaser in enumerate(chasers):
                # create a position array which doesn't include the position of the current chaser
                other_chaser_position_array = np.delete(chaser_position_array, i, axis=0)
                chaser.step(dt=dt_action, neighbors_position=other_chaser_position_array)
                chaser.set_target(t, target.position)

        for chaser in chasers:
            chaser.update(dt)
            if len(blasts) > 0:
                blast_to_remove = []
                for i, blast in enumerate(blasts):
                    blast_position, is_half_life = blast.update(dt)
                    if is_half_life:
                        blast_to_remove.append(i)
                    for blast_position, blast in zip(blast_position, blast.particles_positions_array):
                        blast.set_data(*blast_position)
                # blasts = [blast for i, blast in enumerate(blasts) if i not in blast_to_remove]
            if chaser.distance_to(target) < (chaser.radius + target.radius*1):
                if not chaser.is_shot_confetti:
                    # blasts.append(Blast(ax, chaser.position, chaser.velocity))
                    # blasts[-1].update(dt)
                    chaser.is_shot_confetti = True
                chaser.kill()
            else:
                chaser.done = False
        # remove the chasers that are killed
        for chaser_position, chaser in zip(chaser_positions, chasers):
            # if chaser.done, don't draw the chaser
            if not chaser.done:
                chaser_position.set_data(*chaser.position)
            else:
                chaser_position.set_data(-1000, -1000)
            chaser_position.set_data(*chaser.position)
        # chasers = [chaser for chaser in chasers if not chaser.done]
        # update the target position
        # target.update(dt)
        # update the target position in the chaser
        # chaser.plot()
        # target.plot()
        # # plot the noisy target position
        # plt.scatter(target.position[0] + noise[0], target.position[1] + noise[1], c='r', s=2 * np.pi * target.radius * 100, alpha=0.2)
        # plt.xlim(-bounds, bounds)
        # plt.ylim(-bounds, bounds)
        if (int(t/dt) % (control_to_physics_ratio*5)) == 0:
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
            chaser_speed = np.linalg.norm(chasers[0].velocity)
            # convert the speed to km/h
            # chaser_speed = chaser_speed * 3.6
            plt.title(f"Distance: {chaser.distance_to(target):.2f} m, ww speed: {chaser_speed:.2f} m/s, time: {t:.2f} s")
            plt.pause(0.00001)
        t += dt
    print("time: ", np.round(t,2), "s")