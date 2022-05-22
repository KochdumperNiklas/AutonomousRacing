import numpy as np
import matplotlib.pyplot as plt
from auxiliary.process_lidar_data import process_lidar_data
from auxiliary.adaptive_cruise_control import adaptive_cruise_control
from auxiliary.raceline import *

class PurePersuit:
    """class representing a Pure Persuit controller that follows an optimal raceline"""

    def __init__(self, params, settings):
        """object constructor"""

        # store algorithm settings
        self.settings = settings
        self.params = params

        # load optimal raceline
        self.raceline = load_raceline(settings['path_raceline'])

        # wheelbase of the car
        self.WB = params['lf'] + params['lr']

        # initialization
        self.prev_ind = 0

    def plan(self, x, y, theta, v, scans):
        """compute the control commands based on the observations"""

        # determine lookahead point on the raceline
        lookahead_point = self.get_current_waypoint(x, y, v)

        # compute control commands
        speed, steering_angle = self.get_actuation(x, y, theta, lookahead_point)
        speed = self.settings['VELOCITY_GAIN'] * speed

        if self.settings['ADAPTIVE_CRUISE_CONTROL']:
            speed = adaptive_cruise_control(scans, speed, self.params['width'])

        # visualized planned motion
        if self.settings['VISUALIZE']:
            self.visualization(x, y, theta, lookahead_point, scans)

        return speed, steering_angle
        
    def get_current_waypoint(self, x, y, v):
        """determine the lookahead point toward that the car is steered"""

        # determine closest waypoint on the trajectory
        p = np.array([[x], [y]])
        ind = closest_raceline_point(self.raceline, p, v, 0.01, self.prev_ind)
        self.prev_ind = ind

        # determine lookahead point
        dist = np.linalg.norm(p - self.raceline[0:2, [ind]])

        if dist < self.settings['LOOKAHEAD_DIST']:
            lookahead_point = self.intersection_circle(p, ind)
        elif dist < self.settings['MAX_REACQUIRE']:
            lookahead_point = self.raceline[[0, 1, 2], ind]
        else:
            lookahead_point = np.arrray([[4.0, 0.0, self.raceline[2, ind]]]).T

        return lookahead_point

    def intersection_circle(self, p, ind):
        """determine the point on the raceline that intersects a circle around the current point"""

        # concatenate raceline to avoid problems at the end -> start transition
        raceline = np.concatenate((self.raceline, self.raceline), axis=1)

        # loop over all raceline points
        for i in range(ind, raceline.shape[1]):
            if np.linalg.norm(raceline[0:2, [i]] - p) > self.settings['LOOKAHEAD_DIST']:
                lookahead_point = np.append(raceline[0:2, i], raceline[2, ind])
                break

        return lookahead_point

    def get_actuation(self, x, y, theta, lookahead_point):
        """compute control commands to steer the system towards the lookahead point"""

        # determine speed
        speed = self.settings['VELOCITY_GAIN'] * lookahead_point[2]

        # determine steering angle
        waypoint_y = np.dot(np.array([np.sin(-theta), np.cos(-theta)]), np.expand_dims(lookahead_point[0:2], axis=1)
                            - np.array([[x], [y]]))

        if np.abs(waypoint_y) < 1e-6:
            steering_angle = 0
        else:
            radius = 1/(2.0*waypoint_y[0]/self.settings['LOOKAHEAD_DIST']**2)
            steering_angle = np.arctan(self.WB/radius)

        return speed, steering_angle

    def visualization(self, x, y, theta, lookahead_point, scans):
        """visualize the planned trajectory"""

        lidar_data = process_lidar_data(scans)
        ind = np.where(scans < 60.0)
        lidar_data = lidar_data[:, ind[0]]
        x_min = np.min(lidar_data[0, :])-2
        x_max = np.max(lidar_data[0, :])+2
        y_min = np.min(lidar_data[1, :])-2
        y_max = np.max(lidar_data[1, :])+2
        plt.cla()
        plt.plot(lidar_data[0, :], lidar_data[1, :], '.r', label='lidar measurements')
        rl = self.raceline[0:2, :] - np.array([[x], [y]])
        rl = np.dot(np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]), rl)
        plt.plot(rl[0, :], rl[1, :], 'g', label='optimal raceline')
        v = lookahead_point[2]
        lap = np.dot(np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]),
                     np.expand_dims(lookahead_point[0:2], axis=1) - np.array([[x], [y]]))
        d = lap / np.linalg.norm(lap)
        plt.plot(np.array([0, v*d[0, 0]]), np.array([0, v*d[1, 0]]), 'b', label='velocity vector')
        plt.plot(lap[0], lap[1], '.m', label='lookahead point')
        plt.axis('equal')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.legend(loc='upper right')
        plt.pause(0.01)
