import os.path
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
import alphashape
import cv2
import yaml
from copy import deepcopy
from auxiliary.ScanSimulator import ScanSimulator

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

        # determine a path inside the racetrack
        scanner = ScanSimulator(filepath, yamlpath)
        pose = np.array([x, y, theta])
        orig = -np.asarray(my_dict['origin'][0:2]) #+ (np.asarray(img.shape) * my_dict['resolution'])/2
        line = [np.array([x, y]) + orig]

        """plt.plot(points[0, :], points[1, :], '.r')
        ranges = scanner.scan(pose)
        ranges = ranges[135:-135]
        for j in range(len(ranges)):
            theta_ = deepcopy(theta - np.pi / 2 + np.pi / len(ranges) * j)
            x_ = deepcopy(x + 0.9*ranges[j] * np.cos(theta_))
            y_ = deepcopy(y + 0.9*ranges[j] * np.sin(theta_))
            p_ = np.resize(deepcopy(np.array([x_, y_])) + orig, (2, 1))
            plt.plot([orig[0], p_[0]], [orig[1], p_[1]], 'b')

        plt.show()"""

        for i in range(100):
            ranges = scanner.scan(pose)
            ind = np.argmax(ranges)
            points_ = deepcopy(points - np.array([[x], [y]]) - np.resize(orig, (2, 1)))
            index = np.where(np.sum(points_**2, axis=0) < 25)
            #plt.plot(points[0, :], points[1, :], '.r')
            #plt.plot(points[0, index[0]], points[1, index[0]], '.g')
            #plt.show()
            dist = -np.inf
            for j in range(len(ranges)):
                theta_ = deepcopy(theta - 2.35619449615 + 0.00436332309619 * j)
                x_ = min(ranges[j], 1) * np.cos(theta_)
                y_ = min(ranges[j], 1) * np.sin(theta_)
                dist_ = min(np.sum((points_[:, index[0]] - np.array([[x_], [y_]]))**2, axis=0))
                if dist_ > dist:
                    dist = dist_
                    ind = j
            theta = deepcopy(theta - 2.35619449615 + 0.00436332309619 * ind)
            #tmp = np.dot(np.array([[np.sin(theta), -np.cos(theta)]]), points_)
            #index = np.where(abs(tmp[0]) < 0.2)
            #tmp = np.dot(np.array([[np.cos(theta), np.sin(theta)]]), points_[:, index[0]])
            #index = np.where(tmp[0, :] > 0)
            x = deepcopy(x + min(ranges[ind], 1) * np.cos(theta))
            y = deepcopy(y + min(ranges[ind], 1) * np.sin(theta))
            p = deepcopy(np.array([x, y])) + orig
            line.append(p)
            pose = np.array([x, y, theta])

        line = np.asarray(line)

        # debug
        #img = cv2.resize(img, dsize=(1000, 1000), interpolation=cv2.INTER_CUBIC)

        #alpha = 0.95 * alphashape.optimizealpha(points)
        #hull = alphashape.alphashape(points, 0.5)
        #hull_pts = hull.exterior.coords.xy

        plt.plot(points[0, :], points[1, :], '.r')
        plt.plot(line[:, 0], line[:, 1], 'b')
        plt.plot(line[:, 0], line[:, 1], '.g')
        #plt.plot(points[0, index], points[1, index], '.g')
        plt.show()

        points = []

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i, j] < 1.0:
                    points.append(np.array([i, j]))

        tmp = 1

