import os.path
import numpy as np
from auxiliary.parse_settings import parse_settings
from auxiliary.process_lidar_data import smooth_lidar
from auxiliary.ScanSimulator import ScanSimulator
from algorithms.ManeuverAutomaton import ManeuverAutomaton
from algorithms.MPC_Linear import MPC_Linear
from algorithms.GapFollower import GapFollower
from algorithms.DisparityExtender import DisparityExtender

class SwitchingDriver:
    """class representing a driver that switches between a racing and an obstacle avoidance controller"""

    def __init__(self, params, settings):

        # initialize racing controller
        tmp = parse_settings(settings['CONTROLLER_RACING'], settings['RACETRACK'], settings['VISUALIZE'])
        exec('self.controller_racing = ' + settings['CONTROLLER_RACING'] + '(params, tmp)')

        # initialize obstacle avoidance controller
        tmp = parse_settings(settings['CONTROLLER_OBSTACLE'], settings['RACETRACK'], settings['VISUALIZE'])
        exec('self.controller_obstacles = ' + settings['CONTROLLER_OBSTACLE'] + '(params, tmp)')

        # initialize scan simulator
        racetrack = settings['RACETRACK']
        if racetrack.endswith('Obstacles'):
            racetrack = racetrack[:-9]
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        map_path = os.path.join(parent, 'racetracks', racetrack, racetrack + '.png')
        yaml_path = os.path.join(parent, 'racetracks', racetrack, racetrack + '.yaml')
        self.scanner = ScanSimulator(map_path, yaml_path)

        # initialize object properties
        self.settings = settings
        self.u_prev = np.array([0.0, 0.0])
        self.obstacle = False
        self.switch_count = 0

    def plan(self, x, y, theta, v, scans):
        """compute control commands based on the current observation"""

        # compute control commands for both controllers
        u_racing = self.controller_racing.plan(x, y, theta, v, scans)
        u_obstacle = self.controller_obstacles.plan(x, y, theta, v, scans)

        # check if there is an obstacle or not
        obstacle = self.detect_obstacles(x, y, theta, v, scans)

        # switch controllers (prevent too frequent switching and do not switch during a turn)
        if (obstacle != self.obstacle) and (self.switch_count == 0) and (self.u_prev[1] < self.settings['MAX_STEER']):
            self.obstacle = obstacle
            self.switch_count = 1

        # interpolate between control inputs
        if self.obstacle:
            u1 = u_racing
            u2 = u_obstacle
            str = 'obstacle'
        else:
            u1 = u_obstacle
            u2 = u_racing
            str = 'racing'

        if self.switch_count > 0:
            frac = self.switch_count / self.settings['SWITCH_FRAMES']
            u = np.array([frac * u1[0] + (1 - frac) * u2[0], frac * u1[1] + (1 - frac) * u2[1]])
            self.switch_count += 1
            if self.switch_count == self.settings['SWITCH_FRAMES']+1:
                self.switch_count = 0
        else:
            u = u2
            print(str)

        # store control commands
        self.u_prev = u

        return u

    def detect_obstacles(self, x, y, theta, v, scans):
        """check if there is an obstacle on the track"""

        # compute the expected scan for the current pose based on the map
        pose = np.array([x, y, theta], dtype=float)
        expected_scan = self.scanner.scan(pose)

        # smooth the scan
        expected_smoothed = smooth_lidar(expected_scan, self.settings['MAX_LIDAR_DIST'],
                                         self.settings['PREPROCESS_CONV_SIZE'])
        orig_smoothed = smooth_lidar(scans, self.settings['MAX_LIDAR_DIST'], self.settings['PREPROCESS_CONV_SIZE'])

        # check if there is an obstacle or not
        max_diff = np.max(np.abs(orig_smoothed - expected_smoothed))

        return max_diff > self.settings['MAX_SCAN_DIFF']
