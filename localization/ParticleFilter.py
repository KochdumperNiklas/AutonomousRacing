import os.path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from auxiliary.ScanSimulator import ScanSimulator
from auxiliary.vehicle_model import simulate

class ParticleFilter:
    """class representing the particle filter algorithm for localization"""

    def __init__(self, params, settings, x, y, theta):
        """object constructor"""

        # store settings and parameters
        self.settings = settings
        self.params = params

        # initialize scan simulator
        racetrack = settings['RACETRACK']
        if racetrack.endswith('Obstacles'):
            racetrack = racetrack[:-9]
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        map_path = os.path.join(parent, 'racetracks', racetrack, racetrack + '.png')
        yaml_path = os.path.join(parent, 'racetracks', racetrack, racetrack + '.yaml')
        self.scanner = ScanSimulator(map_path, yaml_path)

        # safe the initial pose of the car
        self.state = [np.array([x, y, 0.0, 0.0, theta, 0.0, 0.0])]

    def localize(self, scans, v, speed, steer):
        """estimate the current pose from the lidar data and the control commands"""

        # estimate pose by simulating the vehicle model
        state = self.state[0]
        state[3] = v
        u = np.array([speed, steer])
        t = np.array([0, 0.01])
        traj = simulate(state, u, t, self.params)

        # generate particles
        traj[-1, 0] = traj[-1, 0]
        traj[-1, 1] = traj[-1, 1]
        traj[-1, 4] = traj[-1, 4]
        particles = [traj[-1, :]]

        for i in range(self.settings['PARTICLES']):
            state_ = deepcopy(traj[-1, :])
            state_[0] = state_[0] + np.random.normal(scale=self.settings['SIGMA_POSITION'])
            state_[1] = state_[1] + np.random.normal(scale=self.settings['SIGMA_POSITION'])
            state_[4] = state_[4] + np.random.normal(scale=self.settings['SIGMA_ORIENTATION'])
            particles.append(state_)

        # select the best particle
        scans = scans[self.settings['LIDAR_RANGE']:-self.settings['LIDAR_RANGE']]
        ind = np.where(scans < self.settings['MAX_LIDAR_DIST'])
        probability = -np.inf

        for p in particles:
            expected_scan = self.scanner.scan(p[[0, 1, 4]])
            expected_scan = expected_scan[self.settings['LIDAR_RANGE']:-self.settings['LIDAR_RANGE']]
            probability_ = -np.sum((expected_scan[ind[0]] - scans[ind[0]])**2)
            if probability_ > probability:
                probability = probability_
                best = deepcopy(p)

        # store the best pose
        self.state.insert(0, deepcopy(best))

        if len(self.state) > self.settings['LENGTH_SMOOTHING']:
            self.state.pop()

        # smoothen the estimation
        if len(self.state) > 1:
            tmp = np.asarray(self.state)
            val = np.expand_dims(np.arange(0, len(self.state)), axis=1)
            x = LinearRegression().fit(val, tmp[:, [0]]).predict([[0]])[0][0]
            y = LinearRegression().fit(val, tmp[:, [1]]).predict([[0]])[0][0]
            theta = LinearRegression().fit(val, tmp[:, [4]]).predict([[0]])[0][0]
        else:
            x = best[0]
            y = best[1]
            theta = best[4]

        return x, y, theta
