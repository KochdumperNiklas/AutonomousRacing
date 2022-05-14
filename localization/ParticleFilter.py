import os.path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from auxiliary.ScanSimulator import ScanSimulator
from auxiliary.vehicle_model import simulate

class ParticleFilter:
    """class representing the particle filter algorithm for localization"""

    def __init__(self, racetrack, x, y, theta, params):
        """object constructor"""

        settings = dict()
        settings['RACETRACK'] = racetrack
        settings['PARTICLES'] = 100
        settings['SIGMA_POSITION'] = 0.1
        settings['SIGMA_ORIENTATION'] = 0.01
        self.settings = settings

        # initialize scan simulator
        racetrack = settings['RACETRACK']
        if racetrack.endswith('Obstacles'):
            racetrack = racetrack[:-9]
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        map_path = os.path.join(parent, 'racetracks', racetrack, racetrack + '.png')
        yaml_path = os.path.join(parent, 'racetracks', racetrack, racetrack + '.yaml')
        self.scanner = ScanSimulator(map_path, yaml_path)

        # safe the initial pose of the car as well as the car parameter
        self.state = np.array([x, y, 0.0, 0.0, theta, 0.0, 0.0])
        self.params = params

    def localize(self, scans, v, speed, steer):
        """estimate the current pose from the lidar data and the control commands"""

        # estimate pose by simulating the vehicle model
        self.state[3] = v
        u = np.array([speed, steer])
        t = np.array([0, 0.01])
        traj = simulate(self.state, u, t, self.params)

        # generate particles
        traj[-1, 0] = traj[-1, 0] + np.random.uniform(-0.1, 0.1)
        traj[-1, 1] = traj[-1, 1] + np.random.uniform(-0.1, 0.1)
        traj[-1, 4] = traj[-1, 4] + np.random.uniform(-0.1, 0.1)
        particles = [traj[-1, :]]

        for i in range(self.settings['PARTICLES']):
            state_ = deepcopy(traj[-1, :])
            state_[0] = state_[0] + np.random.normal(scale=self.settings['SIGMA_POSITION'])
            state_[1] = state_[1] + np.random.normal(scale=self.settings['SIGMA_POSITION'])
            state_[4] = state_[4] + np.random.normal(scale=self.settings['SIGMA_ORIENTATION'])
            particles.append(state_)

        # select the best particle
        probability = -np.inf

        for p in particles:
            expected_scan = self.scanner.scan(p[[0, 1, 4]])
            #plt.plot(expected_scan, 'b')
            probability_ = np.sum(np.exp(-(expected_scan - scans)**2 / (2 * 60**2)))
            probability_ = -np.sum((expected_scan - scans)**2)
            if probability_ > probability:
                probability = probability_
                best = deepcopy(p)

        # store the best pose
        self.state = best

        # debug
        """expected_scan = self.scanner.scan(best[[0, 1, 4]])
        plt.plot(scans, 'r')
        plt.plot(expected_scan, 'b')
        plt.show()"""
        #plt.show()

        return best[0], best[1], best[4]
