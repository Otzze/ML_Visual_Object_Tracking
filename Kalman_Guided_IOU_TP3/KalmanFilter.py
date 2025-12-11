import numpy as np

class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.u = np.array([[u_x], [u_y]])
        self.Xk = np.array([[0], [0], [0], [0]])

        self.A = np.array([[1, 0, dt, 0],
                          [0, 1, 0, dt],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        self.B = np.array([[0.5 * dt**2, 0],
                           [0, 0.5 * dt**2],
                           [dt, 0],
                           [0, dt]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.Q = np.array([[std_acc**2, 0, 0, 0],
                           [0, std_acc**2, 0, 0],
                           [0, 0, std_acc**2, 0],
                           [0, 0, 0, std_acc**2]])

        self.R = np.array([[x_std_meas**2, 0],
                           [0, y_std_meas**2]])

        self.P = np.eye(self.A.shape[0])

    def predict(self):
        self.Xk = np.dot(self.A, self.Xk) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        self.Sk = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.Kk = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.Sk))

        self.Xk = self.Xk + np.dot(self.Kk, (z - np.dot(self.H, self.Xk)))
        self.P = (np.eye(self.A.shape[0]) - np.dot(self.Kk, self.H)).dot(self.P)
