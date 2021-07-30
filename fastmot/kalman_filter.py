from enum import Enum
import numpy as np
import numba as nb

from .utils.rect import get_size


class MeasType(Enum):
    FLOW = 0
    DETECTOR = 1


class KalmanFilter:
    def __init__(self,
                 std_factor_acc=2.25,
                 std_offset_acc=78.5,
                 std_factor_det=(0.08, 0.08),
                 std_factor_klt=(0.14, 0.14),
                 min_std_det=(4.0, 4.0),
                 min_std_klt=(5.0, 5.0),
                 init_pos_weight=5,
                 init_vel_weight=12,
                 vel_coupling=0.6,
                 vel_half_life=2):
        """A simple Kalman filter for tracking bounding boxes in image space.
        The 8-dimensional state space
            x1, y1, x2, y2, v_x1, v_y1, v_x2, v_y2
        contains the bounding box top left corner, bottom right corner,
        and their respective velocities.
        Object motion follows a modified constant velocity model.
        Velocity will decay over time without measurement and bounding box
        corners are coupled together to minimize drifting.

        Parameters
        ----------
        std_factor_acc : float, optional
            Object size scale factor to calculate acceleration standard deviation
            for process noise.
        std_offset_acc : float, optional
            Object size offset to calculate acceleration standard deviation
            for process noise.
        std_factor_det : tuple, optional
            Object width and height scale factors to calculate detector measurement
            noise standard deviation.
        std_factor_klt : tuple, optional
            Object wdith and height scale factors to calculate KLT measurement
            noise standard deviation.
        min_std_det : tuple, optional
            Min detector measurement noise standard deviations.
        min_std_klt : tuple, optional
            Min KLT measurement noise standard deviations.
        init_pos_weight : int, optional
            Scale factor to initialize position state standard deviation.
        init_vel_weight : int, optional
            Scale factor to initialize velocity state standard deviation.
        vel_coupling : float, optional
            Factor to couple bounding box corners.
            Set 0.5 for max coupling and 1.0 to disable coupling.
        vel_half_life : int, optional
            Half life in seconds to decay velocity state.
        """
        assert std_factor_acc >= 0
        self.std_factor_acc = std_factor_acc
        self.std_offset_acc = std_offset_acc
        assert std_factor_det[0] >= 0 and std_factor_det[1] >= 0
        self.std_factor_det = std_factor_det
        assert std_factor_klt[0] >= 0 and std_factor_klt[1] >= 0
        self.std_factor_klt = std_factor_klt
        assert min_std_det[0] >= 0 and min_std_det[1] >= 0
        self.min_std_det = min_std_det
        assert min_std_klt[0] >= 0 and min_std_klt[1] >= 0
        self.min_std_klt = min_std_klt
        assert init_pos_weight >= 0
        self.init_pos_weight = init_pos_weight
        assert init_vel_weight >= 0
        self.init_vel_weight = init_vel_weight
        assert 0 <= vel_coupling <= 1
        self.vel_coupling = vel_coupling
        assert vel_half_life > 0
        self.vel_half_life = vel_half_life

        dt = 1 / 30.
        self.acc_cov, self.meas_mat, self.trans_mat = self._init_mat(dt)

    def reset_dt(self, dt):
        """Resets process noise, measurement and transition matrices from dt.

        Parameters
        ----------
        dt : float
            Time interval in seconds between each frame.
        """
        self.acc_cov, self.meas_mat, self.trans_mat = self._init_mat(dt)

    def create(self, det_meas):
        """Creates Kalman filter state from unassociated measurement.

        Parameters
        ----------
        det_meas : ndarray
            Detected bounding box of [x1, x2, y1, y2].

        Returns
        -------
        ndarray, ndarray
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track.
        """
        mean_pos = det_meas
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        w, h = get_size(det_meas)
        std = np.array([
            max(self.init_pos_weight * self.std_factor_det[0] * w, self.min_std_det[0]),
            max(self.init_pos_weight * self.std_factor_det[1] * h, self.min_std_det[1]),
            max(self.init_pos_weight * self.std_factor_det[0] * w, self.min_std_det[0]),
            max(self.init_pos_weight * self.std_factor_det[1] * h, self.min_std_det[1]),
            max(self.init_vel_weight * self.std_factor_det[0] * w, self.min_std_det[0]),
            max(self.init_vel_weight * self.std_factor_det[1] * h, self.min_std_det[1]),
            max(self.init_vel_weight * self.std_factor_det[0] * w, self.min_std_det[0]),
            max(self.init_vel_weight * self.std_factor_det[1] * h, self.min_std_det[1])
        ], dtype=np.float64)
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Runs Kalman filter prediction step.

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
        ndarray, ndarray
            Returns the mean vector and covariance matrix of the predicted
            state.
        """
        return self._predict(mean, covariance, self.trans_mat, self.acc_cov,
                             self.std_factor_acc, self.std_offset_acc)

    def project(self, mean, covariance, meas_type, multiplier=1.):
        """Projects state distribution to measurement space.

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
        ndarray, ndarray
            Returns the projected mean and covariance matrix of the given state
            estimate.
        """
        if meas_type == MeasType.FLOW:
            std_factor = self.std_factor_klt
            min_std = self.min_std_klt
        elif meas_type == MeasType.DETECTOR:
            std_factor = self.std_factor_det
            min_std = self.min_std_det
        else:
            raise ValueError('Invalid measurement type')

        return self._project(mean, covariance, self.meas_mat, std_factor, min_std, multiplier)

    def update(self, mean, covariance, measurement, meas_type, multiplier=1.):
        """Runs Kalman filter correction step.

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
        ndarray, ndarray
            Returns the measurement-corrected state distribution.
        """
        projected_mean, projected_cov = self.project(mean, covariance, meas_type, multiplier)

        return self._update(mean, covariance, projected_mean,
                            projected_cov, measurement, self.meas_mat)

    def motion_distance(self, mean, covariance, measurements):
        """Computes mahalanobis distance between `measurements` and state distribution.

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
        """Warps kalman filter state using a homography transformation.
        https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1301&context=studentpub

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        H : ndarray
            A 3x3 homography matrix.

        Returns
        -------
        ndarray, ndarray
            Returns the mean vector and covariance matrix of the transformed
            state.
        """
        H1 = np.ascontiguousarray(H[:2, :2])
        h2 = np.ascontiguousarray(H[:2, 2])
        h3 = np.ascontiguousarray(H[2, :2])
        h4 = 1.

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

    def _init_mat(self, dt):
        # acceleration-based process noise
        acc_cov = np.diag([0.25 * dt**4] * 4 + [dt**2] * 4)
        acc_cov[4:, :4] = np.eye(4) * (0.5 * dt**3)
        acc_cov[:4, 4:] = np.eye(4) * (0.5 * dt**3)

        meas_mat = np.eye(4, 8)
        trans_mat = np.eye(8)
        for i in range(4):
            trans_mat[i, i + 4] = self.vel_coupling * dt
            trans_mat[i, (i + 2) % 4 + 4] = (1. - self.vel_coupling) * dt
            trans_mat[i + 4, i + 4] = 0.5**(dt / self.vel_half_life)
        return acc_cov, meas_mat, trans_mat

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _predict(mean, covariance, trans_mat, acc_cov, std_factor_acc, std_offset_acc):
        size = max(get_size(mean[:4])) # max(w, h)
        std = std_factor_acc * size + std_offset_acc
        motion_cov = acc_cov * std**2

        mean = trans_mat @ mean
        covariance = trans_mat @ covariance @ trans_mat.T + motion_cov
        # ensure positive definiteness
        covariance = 0.5 * (covariance + covariance.T)
        return mean, covariance

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _project(mean, covariance, meas_mat, std_factor, min_std, multiplier):
        w, h = get_size(mean[:4])
        std = np.array([
            max(std_factor[0] * w, min_std[0]),
            max(std_factor[1] * h, min_std[1]),
            max(std_factor[0] * w, min_std[0]),
            max(std_factor[1] * h, min_std[1])
        ], dtype=np.float64)
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
        mean = mean + innovation @ kalman_gain.T
        covariance = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return mean, covariance

    @staticmethod
    @nb.njit(fastmath=True, cache=True)
    def _maha_distance(mean, covariance, measurements):
        diff = measurements - mean
        L = np.linalg.cholesky(covariance)
        y = np.linalg.solve(L, diff.T)
        return np.sum(y**2, axis=0)
