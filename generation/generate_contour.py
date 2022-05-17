import os.path
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import yaml
import sys
from copy import deepcopy
from shapely import geometry
sys.path.insert(1, '../')
from auxiliary.ScanSimulator import ScanSimulator
from auxiliary.Line import Line

RACETRACK = 'StonyBrook'

if __name__ == '__main__':
    """generate inner- and outer-boundary of the racetrack and store them in a file"""

    # construct path to required files
    dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    filepath = os.path.join(dirpath, 'racetracks', RACETRACK, RACETRACK + '.png')
    yamlpath = os.path.join(dirpath, 'racetracks', RACETRACK, RACETRACK + '.yaml')
    configpath = os.path.join(dirpath, 'racetracks', RACETRACK, 'config_' + RACETRACK + '.yaml')

    # check if racetrack and required files exist
    if not os.path.exists(filepath):
        msg = "Image file /racetracks/" + RACETRACK + "/" + RACETRACK + ".png defining the racetrack not found"
        raise Exception(msg)

    if not os.path.exists(yamlpath):
        msg = "Config file /racetracks/" + RACETRACK + "/" + RACETRACK + ".yaml not found"
        raise Exception(msg)

    if not os.path.exists(configpath):
        msg = "Config file /racetracks/config_" + RACETRACK + "/" + RACETRACK + ".yaml not found"
        raise Exception(msg)

    # import the image that defines the racetrack
    img = image.imread(filepath)

    # read the image resolution and the racetrack origin from the config file
    with open(yamlpath) as f:
        data = yaml.safe_load(f)

    resolution = data['resolution']
    origin = data['origin']

    # read the starting position for the racetrack from the config file
    with open(configpath) as f:
        data = yaml.safe_load(f)

    x = data['sx']
    y = data['sy']
    theta = data['stheta']

    # convert the image to a point cloud representing the boundary of the racetrack
    indices = np.where(img < 1.0)

    n = len(indices[0])
    points = np.concatenate((np.resize(indices[1], (n, 1)), np.resize(-indices[0], (n, 1))), axis=1)
    points = np.array([[0.0], [img.shape[0]]]) + np.transpose(points)
    points = points * resolution

    # determine centerline of the racetrack
    scanner = ScanSimulator(filepath, yamlpath)
    pose = np.array([x, y, theta])
    orig = -np.asarray(origin[0:2])
    line = [np.array([x, y]) + orig]

    for i in range(1000):
        ranges = scanner.scan(pose)
        points_ = deepcopy(points - np.array([[x], [y]]) - np.resize(orig, (2, 1)))
        index = np.where(np.sum(points_ ** 2, axis=0) < 25)
        dist = -np.inf
        for j in np.arange(135, len(ranges)-135, 10):
            theta_ = deepcopy(theta - 2.35619449615 + 0.00436332309619 * j)
            x_ = min(ranges[j], 1) * np.cos(theta_)
            y_ = min(ranges[j], 1) * np.sin(theta_)
            dist_ = min(np.sum((points_[:, index[0]] - np.array([[x_], [y_]])) ** 2, axis=0))
            if dist_ > dist:
                dist = dist_
                ind = j
        theta = deepcopy(theta - 2.35619449615 + 0.00436332309619 * ind)
        x = deepcopy(x + min(ranges[ind], 1) * np.cos(theta))
        y = deepcopy(y + min(ranges[ind], 1) * np.sin(theta))
        p = deepcopy(np.array([x, y])) + orig
        line.append(p)
        if i > 10 and np.sum((p - line[0]) ** 2) <= 1:
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
    inner_contour = []

    for i in range(line.shape[0]):
        ind = np.argmin(np.sum((points[:, inner] - np.resize(line[i, :], (2, 1))) ** 2, axis=0))
        inner_contour.append(points[:, inner[ind]])

    accuracy = False

    while not accuracy:
        accuracy = True
        for i in range(len(inner_contour)-1):
            p = 0.5*(inner_contour[i] + inner_contour[i+1])
            tmp = np.sum((points[:, inner] - np.resize(p, (2, 1))) ** 2, axis=0)
            ind = np.argmin(tmp)
            if tmp[ind] > 0.2 ** 2:
                inner_contour.insert(i+1, points[:, inner[ind]])
                accuracy = False
                break

    inner_contour = np.asarray(inner_contour) - orig

    # approximate outer-boundary with a polygon
    outer_contour = []

    for i in range(line.shape[0]):
        ind = np.argmin(np.sum((points[:, outer] - np.resize(line[i, :], (2, 1))) ** 2, axis=0))
        outer_contour.append(points[:, outer[ind]])

    accuracy = False

    while not accuracy:
        accuracy = True
        for i in range(len(outer_contour)-1):
            p = 0.5*(outer_contour[i] + outer_contour[i+1])
            tmp = np.sum((points[:, outer] - np.resize(p, (2, 1))) ** 2, axis=0)
            ind = np.argmin(tmp)
            if tmp[ind] > 0.2 ** 2:
                outer_contour.insert(i+1, points[:, outer[ind]])
                accuracy = False
                break

    outer_contour = np.asarray(outer_contour) - orig

    # refine centerline
    line = line[1:-1, :]
    line = np.concatenate((line, line[[0], :]), axis=0)
    distance = np.cumsum(np.sqrt(np.sum(np.diff(line, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    interpolator = interp1d(distance, line, kind='quadratic', axis=0)
    alpha = np.linspace(0, 1, 100)
    line = interpolator(alpha)

    # compute width of the racetrack
    lines_outer = [Line(outer_contour[[0], :].T, outer_contour[[outer_contour.shape[0]-1], :].T)]
    lines_inner = [Line(inner_contour[[0], :].T, inner_contour[[inner_contour.shape[0]-1], :].T)]

    for i in range(outer_contour.shape[0] - 1):
        lines_outer.append(Line(outer_contour[[i], :].T, outer_contour[[i+1], :].T))

    for i in range(inner_contour.shape[0] - 1):
        lines_inner.append(Line(inner_contour[[i], :].T, inner_contour[[i+1], :].T))

    width_right = []
    width_left = []

    rad = np.sqrt(np.max(np.sum(outer_contour**2, axis=1)))
    line = np.concatenate((line[[-1], :], line), axis=0) - np.resize(orig, (1, 2))
    lines_outer.append(lines_outer[0])
    lines_inner.append(lines_inner[0])

    for i in range(1, line.shape[0] - 1):
        d = line[i+1, :] - line[i-1, :]
        d = np.array([[d[1]], [-d[0]]]) / np.linalg.norm(d) * rad
        l = Line(line[[i], :].T - d, line[[i], :].T + d)
        length = np.inf
        for l_ in lines_outer:
            flag, p = l.intersects(l_)
            if flag:
                """for t in lines_outer:
                    t.plot('r')
                plt.plot(line[:, 0], line[:, 1], 'g')
                plt.plot(line[i, 0], line[i, 1], '.r')
                l.plot('b')
                plt.plot(p[0], p[1], '.g')
                plt.show()"""
                length_ = np.linalg.norm(p - line[[i], :].T)
                if length_ < length:
                    length = length_
                    p_ = deepcopy(p)
        del lines_outer[-1]
        lines_outer.append(Line(line[[i], :].T, p_))
        width_right.append(length)

    for i in range(1, line.shape[0] - 1):
        d = line[i+1, :] - line[i-1, :]
        d = np.array([[d[1]], [-d[0]]]) / np.linalg.norm(d) * rad
        l = Line(line[[i], :].T - d, line[[i], :].T + d)
        length = np.inf
        for l_ in lines_inner:
            flag, p = l.intersects(l_)
            if flag:
                """for t in lines_inner:
                    t.plot('r')
                plt.plot(line[:, 0], line[:, 1], 'g')
                plt.plot(line[i, 0], line[i, 1], '.r')
                l.plot('b')
                plt.plot(p[0], p[1], '.g')
                plt.show()"""
                length_ = np.linalg.norm(p - line[[i], :].T)
                if length_ < length:
                    length = length_
                    p_ = deepcopy(p)
        del lines_inner[-1]
        lines_inner.append(Line(line[[i], :].T, p_))
        width_left.append(length)

    width_right = np.expand_dims(np.asarray(width_right), axis=1)
    width_left = np.expand_dims(np.asarray(width_left), axis=1)

    line = line[1:-1, :]
    center = 10*np.concatenate((line, width_left, width_right), axis=1)

    # save contours in files
    path_inner = os.path.join(dirpath, 'racetracks', RACETRACK, 'contour_inner.csv')
    path_outer = os.path.join(dirpath, 'racetracks', RACETRACK, 'contour_outer.csv')
    path_center = os.path.join(dirpath, 'racetracks', RACETRACK, 'centerline.csv')

    np.savetxt(path_inner, inner_contour, delimiter=",")
    np.savetxt(path_outer, outer_contour, delimiter=",")
    np.savetxt(path_center, center, delimiter=",")

    # plot the resulting contours
    plt.plot(points[0, :] - orig[0], points[1, :] - orig[1], '.g')
    plt.plot(inner_contour[:, 0], inner_contour[:, 1], 'r')
    plt.plot(outer_contour[:, 0], outer_contour[:, 1], 'r')
    plt.plot(line[:, 0], line[:, 1], 'b')
    plt.show()
