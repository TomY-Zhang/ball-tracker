# flake8: noqa
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Wrapper class for Kalman Filter
class Ball:
    def __init__(self):
        self.kf = self.get_1d_kf()

    def get_1d_kf(self):
        # initialize Kalman filter
        dt = 1.0                               # time step
        kf = KalmanFilter(dim_x=2, dim_z=1)

        kf.x = np.array([10., 0.])             # initial state
        kf.F = np.array([[1., dt], [0., 1.]])  # state transition matrix
        kf.H = np.array([[1., 0.]])

        # low uncertainty for initial position, high uncertainty for initial velocity
        kf.P *= np.array([[10., 0.],
                          [0., 1000.]])  
        
        kf.R = 10.  # low R value -> trust detections more than model predictions

        return kf
    
    def predict(self):
        self.kf.predict()
    
    def update(self, x):
        self.kf.update(x)

    def get_pos(self):
        return int(self.kf.x[0])