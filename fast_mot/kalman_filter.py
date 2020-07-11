from enum import Enum
from pathlib import Path
import json

import numpy as np
import numba as nb

from .utils import perspectiveTransform, ConfigDecoder


class MeasType(Enum):
    FLOW = 0
    DETECTOR = 1


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        xmin, ymin, xmax, ymax, v_xmin, v_ymin, v_xmax, v_ymax
    contains the bounding box top left corner, bottom right corner,
    and their respective velocities.
    Object motion follows a constant velocity model augmented with velocity 
    coupling and decay for tracking stability.
    """

    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['KalmanFilter']

    def __init__(self, dt, n_init):
        self.dt = dt
        self.n_init = n_init
        self.small_size_std_acc = KalmanFilter.config['small_size_std_acc']
        self.large_size_std_acc = KalmanFilter.config['large_size_std_acc']
        self.min_std_det = KalmanFilter.config['min_std_det']
        self.min_std_flow = KalmanFilter.config['min_std_flow']
        self.std_factor_det = KalmanFilter.config['std_factor_det']
        self.std_factor_flow = KalmanFilter.config['std_factor_flow']
        self.init_std_pos_factor = KalmanFilter.config['init_std_pos_factor']
        self.init_std_vel_factor = KalmanFilter.config['init_std_vel_factor']
        self.vel_coupling = KalmanFilter.config['vel_coupling']
        self.vel_half_life = KalmanFilter.config['vel_half_life']

        self.std_acc_slope = (self.large_size_std_acc[1] - self.small_size_std_acc[1]) / \
                            (self.large_size_std_acc[0] - self.small_size_std_acc[0])
        self.acc_cov = np.diag(np.array([0.25 * self.dt**4] * 4 + [self.dt**2] * 4))
        self.acc_cov[4:, :4] = np.eye(4) * (0.5 * self.dt**3)
        self.acc_cov[:4, 4:] = np.eye(4) * (0.5 * self.dt**3)

        self.meas_mat = np.eye(4, 8)
        self.transition_mat = np.array([
            [1, 0, 0, 0, self.vel_coupling * self.dt, 0, (1 - self.vel_coupling) * self.dt, 0],
            [0, 1, 0, 0, 0, self.vel_coupling * self.dt, 0, (1 - self.vel_coupling) * self.dt], 
            [0, 0, 1, 0, (1 - self.vel_coupling) * self.dt, 0, self.vel_coupling * self.dt, 0], 
            [0, 0, 0, 1, 0, (1 - self.vel_coupling) * self.dt, 0, self.vel_coupling * self.dt], 
            [0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life), 0, 0, 0], 
            [0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life), 0, 0], 
            [0, 0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life), 0],
            [0, 0, 0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life)]
        ])

    def initiate(self, init_meas, flow_meas):
        """Create track from unassociated measurement.
        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.
        """
        center_vel = (flow_meas.center - init_meas.center) / (self.dt * self.n_init)
        mean = np.r_[flow_meas.tlbr, center_vel, center_vel]

        width, height = flow_meas.size
        std = [
            self.init_std_pos_factor * max(width * self.std_factor_flow[0], self.min_std_flow[0]),
            self.init_std_pos_factor * max(height * self.std_factor_flow[1], self.min_std_flow[1]),
            self.init_std_pos_factor * max(width * self.std_factor_flow[0], self.min_std_flow[0]),
            self.init_std_pos_factor * max(height * self.std_factor_flow[1], self.min_std_flow[1]),
            self.init_std_vel_factor * max(width * self.std_factor_flow[0], self.min_std_flow[0]),
            self.init_std_vel_factor * max(height * self.std_factor_flow[1], self.min_std_flow[1]),
            self.init_std_vel_factor * max(width * self.std_factor_flow[0], self.min_std_flow[0]),
            self.init_std_vel_factor * max(height * self.std_factor_flow[1], self.min_std_flow[1]),
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.
        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        return self._predict(mean, covariance, self.small_size_std_acc, self.std_acc_slope, 
            self.acc_cov, self.transition_mat)

    def project(self, mean, covariance, meas_type, conf=1.):
        """Project state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        if meas_type == MeasType.FLOW:
            std_factor = self.std_factor_flow
            min_std = self.min_std_flow
        elif meas_type == MeasType.DETECTOR:
            std_factor = self.std_factor_det
            min_std = self.min_std_det
        else:
            raise ValueError('Invalid measurement type')

        return self._project(mean, covariance, std_factor, min_std, self.meas_mat, conf)

    def update(self, mean, covariance, measurement, meas_type, conf=1.):
        """Run Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance, 
            meas_type, conf)

        return self._update(mean, covariance, projected_mean, 
            projected_cov, measurement, self.meas_mat)

    def motion_distance(self, mean, covariance, measurements):
        """Compute mahalanobis distance between `measurements` and state distribution.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurements : array_like
            An NxM matrix of N samples of dimensionality M.
        Returns
        -------
        ndarray
            Returns a array of size N such that element i
            contains the squared mahalanobis distance for `measurements[i]`.
        """
        projected_mean, projected_cov = self.project(mean, covariance, MeasType.DETECTOR)
        return self._maha_distance(projected_mean, projected_cov, measurements)

    @staticmethod
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def warp(mean, covariance, H_camera):
        pos_tl, pos_br = mean[:2], mean[2:4]
        vel_tl, vel_br = mean[4:6], mean[6:]
        # affine dof
        A = np.ascontiguousarray(H_camera[:2, :2])
        # homography dof
        v = H_camera[2, :2] 
        # translation dof
        t = H_camera[:2, 2] 
        # h33 = H_camera[-1, -1]
        tmp = np.dot(v, pos_tl) + 1
        grad_tl = (tmp * A - np.outer(A @ pos_tl + t, v)) / tmp**2
        tmp = np.dot(v, pos_br) + 1
        grad_br = (tmp * A - np.outer(A @ pos_br + t, v)) / tmp**2

        # warp state
        warped_pos = perspectiveTransform(np.stack((pos_tl, pos_br)) , H_camera)
        mean[:4] = warped_pos.ravel()
        mean[4:6] = grad_tl @ vel_tl
        mean[6:] = grad_br @ vel_br

        # warp covariance too
        for i in nb.prange(0, 4):
            k = 2 * i
            for j in nb.prange(0, 4):
                l = 2 * j
                grad_left = grad_tl if i % 2 == 0 else grad_br
                grad_right = grad_tl if j % 2 == 0 else grad_br
                cov_blk = np.ascontiguousarray(covariance[k:k + 2, l:l + 2])
                covariance[k:k + 2, l:l + 2] = grad_left @ cov_blk @ grad_right.T
        return mean, covariance

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _predict(mean, covariance, small_size_std_acc, std_acc_slope, acc_cov, transition_mat):
        size = max(mean[2:4] - mean[:2] + 1) # max(w, h)
        std_acc = small_size_std_acc[1] + (size - small_size_std_acc[0]) * std_acc_slope
        motion_cov = acc_cov * std_acc**2

        mean = transition_mat @ mean
        covariance = transition_mat @ covariance @ transition_mat.T + motion_cov
        return mean, covariance

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _project(mean, covariance, std_factor, min_std, meas_mat, conf):
        w, h = mean[2:4] - mean[:2] + 1
        std = np.array([
            max(w * std_factor[0], min_std[0]),
            max(h * std_factor[1], min_std[1]),
            max(w * std_factor[0], min_std[0]),
            max(h * std_factor[1], min_std[1])
        ])
        meas_cov = np.diag(np.square(std / conf))

        mean = meas_mat @ mean
        covariance = meas_mat @ covariance @ meas_mat.T
        innovation_cov = covariance + meas_cov
        return mean, innovation_cov

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _update(mean, covariance, proj_mean, proj_cov, measurement, meas_mat):
        kalman_gain = np.linalg.solve(proj_cov, (covariance @ meas_mat.T).T).T
        innovation = measurement - proj_mean
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_covariance

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _maha_distance(mean, covariance, measurements):
        diff = measurements - mean
        L = np.linalg.cholesky(covariance)
        y = np.linalg.solve(L, diff.T)
        return np.sum(y**2, axis=0)

