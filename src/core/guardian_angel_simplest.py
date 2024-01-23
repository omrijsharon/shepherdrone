import matplotlib.pyplot as plt
import numpy as np
from src.utils.actors import Chaser, Target
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

    bounds = 100
    pid = PID(35,0, 0)
    chaser = Chaser(pid=pid, radius=0.05, max_actuator=80, friction_coef=0.04, dim=2)
    target = Target(radius=0.5)

    # target position is random in the top right and left quadrants
    target.reset(np.array([np.random.uniform(-9, 9), np.random.uniform(0, 9)]))
    chaser.reset(np.array([-99, -99]), np.array([0, 0], dtype=np.float64), target.position)

    fig, ax = plt.subplots()
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    target_position, = ax.plot(0, 0, 'ro')
    chaser_position, = ax.plot(0, 0, 'bo')

    # Set up cursor tracker
    tracker = CursorTracker(ax)

    pixel_noise_level = target.radius * 0
    # while the distance between the chaser and the target is greater than the radius of the target + the radius of the chaser
    t = 0
    while chaser.distance_to(target) > (chaser.radius + target.radius):
        # plt.clf()

        target.position = np.array(update_particle_position(target.position, tracker.setpoint, ema_coeff=0.002))
        target_position.set_data(*target.position)

        noise = np.random.normal(0, pixel_noise_level, size=chaser.dim)
        if (int(t/dt) % control_to_physics_ratio) == 0:
            chaser.set_target(t, target.position + noise)
            chaser.step(dt=dt_action)
        chaser.update(dt)
        chaser_position.set_data(*chaser.position)
        # update the target position
        # target.update(dt)
        # update the target position in the chaser
        chaser.set_target(t, target.position)
        # chaser.plot()
        # target.plot()
        # # plot the noisy target position
        # plt.scatter(target.position[0] + noise[0], target.position[1] + noise[1], c='r', s=2 * np.pi * target.radius * 100, alpha=0.2)
        # plt.xlim(-bounds, bounds)
        # plt.ylim(-bounds, bounds)
        if (int(t/dt) % (control_to_physics_ratio*5)) == 0:
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
            chaser_speed = np.linalg.norm(chaser.velocity)
            # convert the speed to km/h
            # chaser_speed = chaser_speed * 3.6
            plt.title(f"Distance: {chaser.distance_to(target):.2f} m, ww speed: {chaser_speed:.2f} m/s, time: {t:.2f} s")
            plt.pause(0.00001)
        t += dt
    print("time: ", np.round(t,2), "s")