import os.path
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import alphashape
import cv2
import yaml
from copy import deepcopy
from shapely import geometry
from auxiliary.ScanSimulator import ScanSimulator
from auxiliary.free_space import linesegment_refinement

class IterativeClosestLine:
    """class representing the iterative-closest-line algorithm for localization"""

    def __init__(self, racetrack, x, y, theta):
        """object constructor"""

        if racetrack.endswith('Obstacles'):
            racetrack = racetrack[:-9]

        inner, outer = self.import_racetrack(racetrack, x, y, theta)

    def import_racetrack(self, racetrack, x, y, theta):
        """import the racetrack as a polygon for the inner- and outer-boundary"""

        # import the image that defines the racetrack
        dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(dirpath, 'racetracks', racetrack, racetrack + '.png')
        yamlpath = os.path.join(dirpath, 'racetracks', racetrack, racetrack + '.yaml')
        img = image.imread(filepath)

        # import the configuration
        with open(yamlpath) as f:
            my_dict = yaml.safe_load(f)

        # convert the image to a point cloud representing the boundary of the racetrack
        indices = np.where(img < 1.0)

        n = len(indices[0])
        points = np.concatenate((np.resize(indices[1], (n, 1)), np.resize(-indices[0], (n, 1))), axis=1)
        points = np.array([[0.0], [img.shape[1]]]) + np.transpose(points)
        points = points * my_dict['resolution']

        # determine centerline of the racetrack
        scanner = ScanSimulator(filepath, yamlpath)
        pose = np.array([x, y, theta])
        orig = -np.asarray(my_dict['origin'][0:2])
        line = [np.array([x, y]) + orig]

        for i in range(1000):
            ranges = scanner.scan(pose)
            ind = np.argmax(ranges)
            points_ = deepcopy(points - np.array([[x], [y]]) - np.resize(orig, (2, 1)))
            index = np.where(np.sum(points_**2, axis=0) < 25)
            dist = -np.inf
            for j in np.arange(0, len(ranges), 20):
                theta_ = deepcopy(theta - 2.35619449615 + 0.00436332309619 * j)
                x_ = min(ranges[j], 1) * np.cos(theta_)
                y_ = min(ranges[j], 1) * np.sin(theta_)
                dist_ = min(np.sum((points_[:, index[0]] - np.array([[x_], [y_]]))**2, axis=0))
                if dist_ > dist:
                    dist = dist_
                    ind = j
            theta = deepcopy(theta - 2.35619449615 + 0.00436332309619 * ind)
            x = deepcopy(x + min(ranges[ind], 1) * np.cos(theta))
            y = deepcopy(y + min(ranges[ind], 1) * np.sin(theta))
            p = deepcopy(np.array([x, y])) + orig
            line.append(p)
            if i > 10 and np.sum((p - line[0])**2) <= 1:
                break
            pose = np.array([x, y, theta])

        line = np.asarray(line)

        # convert center-line to polygon
        pgon = geometry.Polygon([*zip(line[:, 0], line[:, 1])])

        # divide points into inside and outside boundary of the racetrack
        inner = []
        outer = []

        for i in range(points.shape[1]):
            if pgon.contains(geometry.Point(points[0, i], points[1, i])):
                inner.append(i)
            else:
                outer.append(i)

        # approximate inner boundary with a polygon
        inner_contur = []

        for i in range(line.shape[0]):
            ind = np.argmin(np.sum((points[:, inner] - np.resize(line[i, :], (2, 1)))**2, axis=0))
            inner_contur.append(points[:, inner[ind]])

        inner_contur = np.asarray(inner_contur)
        #test = linesegment_refinement(points[:, inner])

        plt.plot(points[0, :], points[1, :], '.r')
        plt.plot(line[:, 0], line[:, 1], 'b')
        plt.plot(line[:, 0], line[:, 1], '.g')
        plt.plot(points[0, inner], points[1, inner], '.k')
        plt.plot(inner_contur[:, 0], inner_contur[:, 1], 'm')
        #plt.plot(points[0, index], points[1, index], '.g')
        plt.show()

        points = []

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] < 1.0:
                    points.append(np.array([i, j]))

        tmp = 1

