import numpy as np

class PID:
    # pid class with FF and anti-windup.
    # FF is a feedforward term that is added to the output before the PID terms are added.
    # anti-windup is a term that is added to the output after the PID terms are added.
    # anti-windup is the integral term multiplied by the anti-windup gain.
    # the derivative part is calculated using the current measurement and the previous measurement.
    def __init__(self, kp, ki, kd, ff=0, min_output=-np.inf, max_output=np.inf):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ff = ff
        self.min_output = min_output
        self.max_output = max_output

        self.integral = 0
        self.prev_error = 0
        self.prev_measurement = 0
        self.prev_target = 0
        self.anti_windup_gate = False

    def reset(self, measurement, target):
        self.integral = 0
        self.prev_error = np.linalg.norm(target - measurement)
        self.prev_measurement = measurement
        self.prev_target = target
        self.anti_windup_gate = False

    # def update(self, measurement, target, dt):
    #     error = target - measurement
    #     if not self.anti_windup_gate:
    #         self.integral += error * dt
    #     derivative = -(measurement - self.prev_measurement) / dt
    #     output = self.ff + self.kp * error + self.ki * self.integral + self.kd * derivative
    #     # if output is saturated, add anti-windup
    #     is_saturated = output <= self.min_output or output >= self.max_output
    #     if is_saturated and np.sign(self.integral) == np.sign(error):
    #         self.anti_windup_gate = True
    #     else:
    #         self.anti_windup_gate = False
    #     output = np.clip(output, self.min_output, self.max_output)
    #     self.prev_error = error
    #     self.prev_measurement = measurement
    #     self.prev_target = target
    #     return output
    def update(self, error, dt):
        # error = target - measurement
        if not self.anti_windup_gate:
            self.integral += error * dt
        # derivative = -(measurement - self.prev_measurement) / dt
        derivative = (error - self.prev_error) / dt
        output = self.ff + self.kp * error + self.ki * self.integral + self.kd * derivative
        # if output is saturated, add anti-windup
        is_saturated = output <= self.min_output or output >= self.max_output
        if is_saturated and np.sign(self.integral) == np.sign(error):
            self.anti_windup_gate = True
        else:
            self.anti_windup_gate = False
        output = np.clip(output, self.min_output, self.max_output)
        self.prev_error = error
        # self.prev_measurement = measurement
        # self.prev_target = target
        return output

    def set_output_limits(self, min_output, max_output):
        self.min_output = min_output
        self.max_output = max_output


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from time import sleep
    pid = PID(kp=0.1, ki=0.1, kd=0.01, ff=0, min_output=-1, max_output=1)
    pid.reset(0, 0)
    output = []
    integral_term = []
    for i in range(100):
        output.append(pid.update(0, 1, dt=1e-1))
        integral_term.append(pid.integral)
        sleep(0.1)
    plt.plot(output)
    plt.plot(integral_term)
    plt.show()