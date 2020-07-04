from enum import Enum
from pathlib import Path
import json

import numpy as np
import numba as nb
import scipy.linalg

from .utils import perspectiveTransform, ConfigDecoder


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x, y, a, h, vx, vy, va, vh
    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """
    class Meas(Enum):
        FLOW = 0
        CNN = 1

    with open(Path(__file__).parent / 'configs' / 'mot.json') as config_file:
        config = json.load(config_file, cls=ConfigDecoder)['KalmanFilter']

    def __init__(self, dt):
        self.dt = dt
        self.n_init = KalmanFilter.config['n_init']
        self.small_size_std_acc = KalmanFilter.config['small_size_std_acc']
        self.large_size_std_acc = KalmanFilter.config['large_size_std_acc']
        self.min_std_cnn = KalmanFilter.config['min_std_cnn']
        self.min_std_flow = KalmanFilter.config['min_std_flow']
        self.std_factor_cnn = KalmanFilter.config['std_factor_cnn']
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

        # # Create Kalman filter model matrices.
        # self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        # for i in range(ndim):
        #     self._motion_mat[i, ndim + i] = dt
        # self._update_mat = np.eye(ndim, 2 * ndim)

        # # Motion and observation uncertainty are chosen relative to the current
        # # state estimate. These weights control the amount of uncertainty in
        # # the model. This is a bit hacky.
        # self._std_weight_position = 1. / 20
        # self._std_weight_velocity = 1. / 160

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

        # mean_pos = measurement
        # mean_vel = np.zeros_like(mean_pos)
        # mean = np.r_[mean_pos, mean_vel]

        # std = [
        #     2 * self._std_weight_position * measurement[3],
        #     2 * self._std_weight_position * measurement[3],
        #     1e-2,
        #     2 * self._std_weight_position * measurement[3],
        #     10 * self._std_weight_velocity * measurement[3],
        #     10 * self._std_weight_velocity * measurement[3],
        #     1e-5,
        #     10 * self._std_weight_velocity * measurement[3]]
        # covariance = np.diag(np.square(std))
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
        size = np.max(mean[:2] - mean[2:4] + 1) # max(w, h)
        std_acc = self.small_size_std_acc[1] + (size - self.small_size_std_acc[0]) * self.std_acc_slope
        motion_cov = self.acc_cov * std_acc**2

        mean = self.transition_mat @ mean
        covariance = np.linalg.multi_dot((
            self.transition_mat, covariance, self.transition_mat.T)) + motion_cov

        # std_pos = [
        #     self._std_weight_position * mean[3],
        #     self._std_weight_position * mean[3],
        #     1e-2,
        #     self._std_weight_position * mean[3]]
        # std_vel = [
        #     self._std_weight_velocity * mean[3],
        #     self._std_weight_velocity * mean[3],
        #     1e-5,
        #     self._std_weight_velocity * mean[3]]
        # motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # mean = np.dot(self._motion_mat, mean)
        # covariance = np.linalg.multi_dot((
        #     self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

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
        if meas_type == KalmanFilter.Meas.FLOW:
            std_factor = self.std_factor_flow
            min_std = self.min_std_flow
        elif meas_type == KalmanFilter.Meas.CNN:
            std_factor = self.std_factor_cnn
            min_std = self.min_std_cnn
        else:
            raise ValueError('Invalid measurement type')

        w, h = mean[:2] - mean[2:4] + 1
        std = [
            max(w * std_factor[0], min_std[0]),
            max(h * std_factor[1], min_std[1]),
            max(w * std_factor[0], min_std[0]),
            max(h * std_factor[1], min_std[1])
        ]
        innovation_cov = np.diag(np.square(std / conf))

        mean = self.meas_mat @ mean
        covariance = np.linalg.multi_dot((
            self.meas_mat, covariance, self.meas_mat.T))
        return mean, covariance + innovation_cov

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

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self.transition_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

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
        projected_mean, projected_cov = self.project(mean, covariance, KalmanFilter.Meas.CNN)

        diff = measurements - projected_mean
        L = np.linalg.cholesky(projected_cov)
        y = scipy.linalg.solve_triangular(L, diff.T, lower=True, overwrite_b=True, check_finite=False)
        return np.sum(y**2, axis=0)

    def warp(self, mean, covariance, H_camera):
        pos_tl, pos_br = mean[:2], mean[2:4]
        vel_tl, vel_br = mean[4:6], mean[6:]
        # affine dof
        A = H_camera[:2, :2]
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
        warped_pos = perspectiveTransform(np.stack([pos_tl, pos_br]) , H_camera)
        mean[:4] = warped_pos.ravel()
        mean[4:6] = grad_tl @ vel_tl
        mean[6:] = grad_br @ vel_br

        # warp covariance too
        for i in range(0, 8, 2):
            for j in range(0, 8, 2):
                grad_left = grad_tl if i // 2 % 2 == 0 else grad_br
                grad_right = grad_tl if j // 2 % 2 == 0 else grad_br
                covariance[i:i + 2, j:j + 2] = \
                    np.linalg.multi_dot([grad_left, covariance[i:i + 2, j:j + 2], grad_right.T])
        return mean, covariance
