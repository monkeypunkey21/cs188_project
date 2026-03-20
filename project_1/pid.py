import numpy as np

class PID:
    def __init__(self, kp, ki, kd, target):
        # Initialize gains and setpoint
        self.kp = kp
        self.ki = ki
        self.kd = kd

        target_dim = target.shape[0]
        self.target = target
        self.integral = np.zeros(target_dim)
        self.previous_error = np.zeros(target_dim)

        self.initialized = False
        
    def reset(self, target=None):
        # Reset internal error history
        self.previous_error = np.zeros_like(self.previous_error)
        self.integral = np.zeros_like(self.integral)


        if target is not None:
            self.target = target
        
    def get_error(self):
        # Return magnitude of last error
        return np.linalg.norm(self.previous_error) 

    def update(self, current_pos, dt):
        # Compute and return control signal
        
        error = self.target - current_pos
        self.integral += error * dt

        P = self.kp * error
        I = self.ki * self.integral
        D = self.kd * (error - self.previous_error) / dt if dt > 0 else np.zeros_like(error)

        control_signal = P + I + D

        self.previous_error = error

        return control_signal
