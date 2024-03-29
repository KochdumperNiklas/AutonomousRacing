import os.path
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import alphashape
import cv2
import yaml
import math
import pandas as pd
from copy import deepcopy
from shapely import geometry
from auxiliary.ScanSimulator import ScanSimulator
from auxiliary.free_space import linesegment_refinement
from auxiliary.Line import Line
from auxiliary.Polytope import Polytope
from auxiliary.vehicle_model import simulate
from auxiliary.process_lidar_data import process_lidar_data
from auxiliary.free_space import linesegment_refinement

class IterativeClosestLine:
    """class representing the iterative-closest-line algorithm for localization"""

    def __init__(self, params, settings, x, y, theta):
        """object constructor"""

        # store settings and parameters
        self.settings = settings
        self.params = params

        # load the map of the racetrack
        racetrack = self.settings['RACETRACK']
        if racetrack.endswith('Obstacles'):
            racetrack = racetrack[:-9]

        self.map, self.pgon_inner, self.pgon_outer, self.map_centers = self.import_racetrack(racetrack)

        # safe the initial pose of the car as well as the car parameter
        self.x = x
        self.y = y
        self.theta = theta

    def localize(self, scans, v, speed, steer):
        """estimate the current pose from the lidar data and the control commands"""

        # estimate pose by simulating the vehicle model
        x0 = np.array([self.x, self.y, 0.0, v, self.theta, 0.0, 0.0])
        u = np.array([speed, steer])
        t = np.array([0, 0.01])
        traj = simulate(x0, u, t, self.params)
        self.x = traj[-1, 0] #+ np.random.uniform(-0.1, 0.1)
        self.y = traj[-1, 1] #+ np.random.uniform(-0.1, 0.1)
        self.theta = traj[-1, 4] #+ np.random.uniform(-0.05, 0.05)

        # convert the lidar data to line segments
        points = process_lidar_data(scans)
        ind = np.where(scans < self.settings['MAX_LIDAR_DIST'])
        tmp = linesegment_refinement(points[:, ind[0]])

        segments = []
        for s in tmp:
            if s.shape[1] > self.settings['MIN_POINTS']:
                l = Line(s[:, [0]], s[:, [s.shape[1] - 1]])
                if min(np.sqrt(np.sum((s - l.center())**2, axis=0))) < self.settings['EMPTY_DIST']:
                    segments.append(l)

        # determine relevant map segments that have to be considered
        C = np.concatenate((np.identity(2), -np.identity(2)), axis=0)
        d = np.array([[np.max(points[0, :]) + 2], [np.max(points[1, :]) + 2],
                      [-np.min(points[0, :]) + 2], [-np.min(points[1, :]) + 2]])
        poly = Polytope(C, d)
        poly.rotate(self.theta)
        poly.shift(np.array([[self.x], [self.y]]))

        ind = np.where(np.max(np.dot(poly.c, self.map_centers) - poly.d, axis=0) <= 0)
        map = [self.map[i] for i in ind[0]]

        # update pose estimation with iterative closest line algorithm
        if len(map) > 0:
            x1, y1, theta1 = self.iterative_closest_line(segments, map, self.x, self.y, self.theta)

            if np.sqrt((x1-self.x)**2 + (y1-self.y)**2) < self.settings['MAX_DIST_POSITION'] \
                    and abs(self.theta - theta1) < self.settings['MAX_DIST_ORIENTATION'] \
                    and not self.pgon_inner.contains(geometry.Point(x1, y1)) \
                    and self.pgon_outer.contains(geometry.Point(x1, y1)):
                self.x = x1
                self.y = y1
                self.theta = theta1

        # project estimated point back onto the racetrack if it is outside
        if self.pgon_inner.contains(geometry.Point(self.x, self.y)) or \
                not self.pgon_outer.contains(geometry.Point(self.x, self.y)):
            dist = np.inf
            for m in map:
                p_, dist_ = self.closest_point_line(m, np.array([[self.x], [self.y]]))
                if dist_ < dist:
                    p = p_
                    dist = dist_
            self.x = p[0][0]
            self.y = p[1][0]

        return self.x, self.y, self.theta

    def iterative_closest_line(self, segments, map, x, y, theta):
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
            for j in range(len(map)):
                _, dist_ = self.closest_point_line(map[j], c)
                if dist_ < dist:
                    dirs[:, [i]] = map[j].d
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
            p = map[ind[i]]
            s = (deepcopy(segments[i]) * R + o).center()
            tmp, _ = self.closest_point_line(p, s)
            o_ = o_ + segments[i].len * (tmp - s)
            length += segments[i].len

        o = o + 1 / length * o_

        # extract pose from rotation matrix and offset
        x = o[0][0]
        y = o[1][0]
        theta = math.atan2(R[1, 0], R[0, 0])

        # debug
        #for s in segments:
        #    (s * R + o).plot('r')
        #for m in map:
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

        # generate polygons that represent the inside and the outside of the racetrack
        pgon_inside = geometry.Polygon([*zip(inner_contour[0, :], inner_contour[1, :])])
        pgon_outside = geometry.Polygon([*zip(outer_contour[0, :], outer_contour[1, :])])

        # store centers for the line segments of the map
        cen = []

        for m in map:
            cen.append(np.resize(m.center(), (2, )))

        cen = np.transpose(np.asarray(cen))

        return map, pgon_inside, pgon_outside, cen
