import os.path
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import alphashape
import cv2
import yaml
import pandas as pd
from copy import deepcopy
from shapely import geometry
from auxiliary.ScanSimulator import ScanSimulator
from auxiliary.free_space import linesegment_refinement
from auxiliary.Line import Line
from auxiliary.vehicle_model import simulate
from auxiliary.process_lidar_data import process_lidar_data
from auxiliary.free_space import linesegment_refinement

class IterativeClosestLine:
    """class representing the iterative-closest-line algorithm for localization"""

    def __init__(self, racetrack, x, y, theta, params):
        """object constructor"""

        # load the map of the racetrack
        if racetrack.endswith('Obstacles'):
            racetrack = racetrack[:-9]

        self.map = self.import_racetrack(racetrack)

        # safe the initial pose of the car as well as the car parameter
        self.x = x
        self.y = y
        self.theta = theta

        self.params = params

    def localize(self, scans, v, speed, steer):
        """estimate the current pose from the lidar data and the control commands"""

        # estimate pose by simulating the vehicle model
        x0 = np.array([self.x, self.y, 0.0, v, self.theta, 0.0, 0.0])
        u = np.array([speed, steer])
        t = np.array([0, 0.01])
        traj = simulate(x0, u, t, self.params)
        x_ = traj[-1, 0] + np.random.uniform(-0.1, 0.1)
        y_ = traj[-1, 1] + np.random.uniform(-0.1, 0.1)
        theta_ = traj[-1, 4] + np.random.uniform(-0.05, 0.05)

        # convert the lidar data to line segments
        points = process_lidar_data(scans)
        tmp = linesegment_refinement(points)

        segments = []
        for s in tmp:
            segments.append(Line(s[:, [0]], s[:, [s.shape[1] - 1]]))

        # update pose estimation with iterative closest line algorithm
        for i in range(2):
            x_, y_, theta_ = self.iterative_closest_line(segments, x_, y_, theta_)
        self.x = x_
        self.y = y_
        self.theta = theta_

        return self.x, self.y, self.theta

    def iterative_closest_line(self, segments, x, y, theta):
        # iterative closest line refinement for pattern matching

        # compute current rotation matrix and shift
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        o = np.array([[x], [y]])

        # apply current state space transformation measurements
        segments_ = []

        for i in range(len(segments)):
            segments_.append(segments[i] * R + o)

        # determine closest line segment in the pattern for each line segment
        dirs = np.zeros((2, len(segments_)))
        ind = np.zeros((len(segments_),)).astype(int)

        for i in range(len(segments_)):
            dist = np.inf
            c = segments_[i].center()
            for j in range(len(self.map)):
                _, dist_ = self.closest_point_line(self.map[j], c)
                if dist_ < dist:
                    dirs[:, [i]] = self.map[j].d
                    ind[i] = j
                    dist = dist_

        # compute covariance matrix
        W = np.zeros((2, 2))

        for i in range(len(segments_)):
            if np.dot(np.transpose(segments_[i].d), dirs[:, [i]]) > 0:
                W = W + segments_[i].len * np.dot(segments_[i].d, np.transpose(dirs[:, [i]]))
            else:
                W = W + segments_[i].len * np.dot(segments_[i].d, -np.transpose(dirs[:, [i]]))

        # update rotation matrix using a singular value decomposition of the covariance matrix
        U, _, V = np.linalg.svd(W)
        R_ = np.dot(U, np.transpose(V))
        R = deepcopy(np.dot(np.transpose(R_), R))

        # update rotation using average distance to closest point in the template
        o_ = np.zeros((2, 1))
        length = 0.0

        for i in range(len(segments)):
            p = self.map[ind[i]]
            s = (deepcopy(segments[i]) * R + o).center()
            tmp, _ = self.closest_point_line(p, s)
            o_ = o_ + segments[i].len * (tmp - s)
            length += segments[i].len

        o = o + 1 / length * o_

        # extract pose from rotation matrix and offset
        x = o[0][0]
        y = o[1][0]
        theta = np.arccos(R[0, 0])

        # debug
        #for s in segments:
        #    (s * R + o).plot('r')
        #for m in self.map:
        #    m.plot('b')
        #plt.show()

        return x, y, theta

    def closest_point_line(self, line, p):
        """determines which point on the line is closest to the point p"""

        tmp = np.dot(np.transpose(line.d), p - line.p1)
        if tmp > 0 and tmp < line.len:
            p_ = line.p1 + tmp * line.d
            dist = np.linalg.norm(p_ - p)
            return p_, dist
        else:
            points = np.concatenate((line.p1, line.p2), axis=1)
            dist = np.linalg.norm(points - p, axis=0)
            ind = np.argmin(dist)
            return points[:, [ind]], dist[ind]

    def import_racetrack(self, racetrack):
        """import the racetrack as a polygon for the inner- and outer-boundary"""

        # import contours of the racetrack boundaries
        dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path_inner = os.path.join(dirpath, 'racetracks', racetrack, 'contour_inner.csv')
        path_outer = os.path.join(dirpath, 'racetracks', racetrack, 'contour_outer.csv')

        if not os.path.exists(path_inner) or not os.path.exists(path_outer):
            msg = "Files storing the inner- and outer-contour do not exist for this racetrack \n"
            msg += "Execute the file /generation/generate_contour.py to generate the contour."
            raise Exception(msg)

        inner_contour = np.transpose(np.asarray(pd.read_csv(path_inner)))
        outer_contour = np.transpose(np.asarray(pd.read_csv(path_outer)))

        # convert contours to a list of lines that defines the racetrack
        map = [Line(inner_contour[:, 0], inner_contour[:, -1])]

        for i in range(inner_contour.shape[1]-1):
            map.append(Line(inner_contour[:, i], inner_contour[:, i+1]))

        map.append(Line(outer_contour[:, 0], outer_contour[:, -1]))

        for i in range(outer_contour.shape[1]-1):
            map.append(Line(outer_contour[:, i], outer_contour[:, i+1]))

        return map
