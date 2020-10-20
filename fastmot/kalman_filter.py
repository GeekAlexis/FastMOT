from enum import Enum
import numpy as np
import numba as nb

from .utils.rect import get_size


class MeasType(Enum):
    FLOW = 0
    DETECTOR = 1


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    The 8-dimensional state space
        x1, y1, x2, y2, v_x1, v_y1, v_x2, v_y2
    contains the bounding box top left corner, bottom right corner,
    and their respective velocities.
    Object motion follows a modified constant velocity model.
    Velocity will decay over time without measurement and bounding box
    corners are coupled together to minimize drifting.
    Parameters
    ----------
    dt : float
        Time interval in seconds between each frame.
    config : Dict
        Kalman Filter hyperparameters.
    """

    def __init__(self, dt, config):
        self.dt = dt
        self.small_std_acc = config['small_std_acc']
        self.large_std_acc = config['large_std_acc']
        self.min_std_det = config['min_std_det']
        self.min_std_flow = config['min_std_flow']
        self.std_factor_det = config['std_factor_det']
        self.std_factor_flow = config['std_factor_flow']
        self.init_pos_std_factor = config['init_pos_std_factor']
        self.init_vel_std_factor = config['init_vel_std_factor']
        self.vel_coupling = config['vel_coupling']
        self.vel_half_life = config['vel_half_life']

        # acceleration std adjustment rate with respect to pixel width/height
        self.std_acc_rate = ((self.large_std_acc[1] - self.small_std_acc[1]) /
                             (self.large_std_acc[0] - self.small_std_acc[0]))

        # acceleration-based process noise
        self.acc_cov = np.diag(np.array([0.25 * self.dt**4] * 4 + [self.dt**2] * 4, dtype=np.float))
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
            [0, 0, 0, 0, 0, 0, 0, 0.5**(self.dt / self.vel_half_life)],
        ], dtype=np.float)

    def initiate(self, det_meas):
        """
        Creates track from unassociated measurement.
        Parameters
        ----------
        det_meas : ndarray
            Detected bounding box of [x1, x2, y1, y2].
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track.
        """
        mean_pos = det_meas
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        width, height = get_size(det_meas)
        std = np.array([
            self.init_pos_std_factor * max(width * self.std_factor_det[0], self.std_factor_det[0]),
            self.init_pos_std_factor * max(height * self.std_factor_det[1], self.std_factor_det[1]),
            self.init_pos_std_factor * max(width * self.std_factor_det[0], self.std_factor_det[0]),
            self.init_pos_std_factor * max(height * self.std_factor_det[1], self.std_factor_det[1]),
            self.init_vel_std_factor * max(width * self.std_factor_det[0], self.std_factor_det[0]),
            self.init_vel_std_factor * max(height * self.std_factor_det[1], self.std_factor_det[1]),
            self.init_vel_std_factor * max(width * self.std_factor_det[0], self.std_factor_det[0]),
            self.init_vel_std_factor * max(height * self.std_factor_det[1], self.std_factor_det[1]),
        ], dtype=np.float)
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Runs Kalman filter prediction step.
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
            state.
        """
        return self._predict(mean, covariance, self.small_std_acc, self.std_acc_rate,
                             self.acc_cov, self.transition_mat)

    def project(self, mean, covariance, meas_type, multiplier=1.):
        """
        Projects state distribution to measurement space.
        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        meas_type : MeasType
            Measurement type indicating where the measurement comes from.
        multiplier : float
            Multiplier used to adjust the measurement std.
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

        return self._project(mean, covariance, std_factor, min_std, self.meas_mat, multiplier)

    def update(self, mean, covariance, measurement, meas_type, multiplier=1.):
        """
        Runs Kalman filter correction step.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            Bounding box of [x1, x2, y1, y2].
        meas_type : MeasType
            Measurement type indicating where the measurement comes from.
        multiplier : float
            Multiplier used to adjust the measurement std.
        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance, meas_type, multiplier)

        return self._update(mean, covariance, projected_mean,
                            projected_cov, measurement, self.meas_mat)

    def motion_distance(self, mean, covariance, measurements):
        """
        Computes mahalanobis distance between `measurements` and state distribution.
        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurements : array_like
            An Nx4 matrix of N samples of [x1, x2, y1, y2].
        Returns
        -------
        ndarray
            Returns a array of size N such that element i
            contains the squared mahalanobis distance for `measurements[i]`.
        """
        projected_mean, projected_cov = self.project(mean, covariance, MeasType.DETECTOR)
        return self._maha_distance(projected_mean, projected_cov, measurements)

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def warp(mean, covariance, H):
        """
        Warps kalman filter state using a homography transformation.
        https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1301&context=studentpub
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        H : ndarray
            A 3x3 homography matrix.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the transformed
            state.
        """
        H1 = np.ascontiguousarray(H[:2, :2])
        h2 = np.ascontiguousarray(H[:2, 2])
        h3 = np.ascontiguousarray(H[2, :2])
        h4 = 1

        E1 = np.eye(8, 2)
        E3 = np.eye(8, 2, -4)
        M = E1 @ H1 @ E1.T + E3 @ H1 @ E3.T
        M31 = E3 @ H1 @ E1.T
        w12 = E1 @ h2
        w13 = E1 @ h3
        w33 = E3 @ h3
        u = M @ mean + w12
        v = M31 @ mean + E3 @ h2
        a = np.dot(w13, mean) + h4
        b = np.dot(w33, mean)
        # transform top left mean
        mean_tl = u / a - b * v / a**2
        # compute top left Jacobian
        F_tl = M / a - (np.outer(u, w13) + b * M31 + np.outer(v, w33)) / a**2 + \
            (2 * b * np.outer(v, w13)) / a**3

        E2 = np.eye(8, 2, -2)
        E4 = np.eye(8, 2, -6)
        M = E2 @ H1 @ E2.T + E4 @ H1 @ E4.T
        M42 = E4 @ H1 @ E2.T
        w22 = E2 @ h2
        w23 = E2 @ h3
        w43 = E4 @ h3
        u = M @ mean + w22
        v = M42 @ mean + E4 @ h2
        a = np.dot(w23, mean) + h4
        b = np.dot(w43, mean)
        # transform bottom right mean
        mean_br = u / a - b * v / a**2
        # compute bottom right Jacobian
        F_br = M / a - (np.outer(u, w23) + b * M42 + np.outer(v, w43)) / a**2 + \
            (2 * b * np.outer(v, w23)) / a**3

        # add them together
        mean = mean_tl + mean_br
        F = F_tl + F_br
        # tranform covariance with Jacobian
        covariance = F @ covariance @ F.T
        return mean, covariance

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _predict(mean, covariance, small_std_acc, std_acc_rate, acc_cov, transition_mat):
        size = max(mean[2:4] - mean[:2] + 1) # max(w, h)
        std_acc = small_std_acc[1] + (size - small_std_acc[0]) * std_acc_rate
        motion_cov = acc_cov * std_acc**2

        mean = transition_mat @ mean
        covariance = transition_mat @ covariance @ transition_mat.T + motion_cov
        return mean, covariance

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _project(mean, covariance, std_factor, min_std, meas_mat, multiplier):
        w, h = mean[2:4] - mean[:2] + 1
        std = np.array([
            max(w * std_factor[0], min_std[0]),
            max(h * std_factor[1], min_std[1]),
            max(w * std_factor[0], min_std[0]),
            max(h * std_factor[1], min_std[1])
        ])
        meas_cov = np.diag(np.square(std * multiplier))

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
