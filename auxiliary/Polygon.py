import numpy as np
import math
from shapely import geometry
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io import loadmat
from auxiliary.Polytope import Polytope

class Polygon:
    """class represnting a polygon"""

    def __init__(self, x, y):
        """class constructor"""

        x = np.resize(x, (len(x), 1))
        y = np.resize(y, (len(y), 1))
        self.vertices = np.concatenate((x, y), axis=1)

    def __and__(self, set):
        """intersection of two polygons"""

        # compute intersection
        pgon1 = geometry.Polygon([*zip(self.vertices[:, 0], self.vertices[:, 1])])
        pgon2 = geometry.Polygon([*zip(set.vertices[:, 0], set.vertices[:, 1])])

        if not pgon1.is_valid:
            pgon1 = pgon1.buffer(0)

        if not pgon2.is_valid:
            pgon2 = pgon2.buffer(0)

        pgon = pgon1.intersection(pgon2)

        # select largest region if there are multiple disjoint regions
        if isinstance(pgon, geometry.MultiPolygon):
            area = -math.inf
            for i in range(len(pgon)):
                if pgon[i].area > area:
                    tmp = pgon[i]

            pgon = tmp

        # remove redundant vertices
        x, y = pgon.exterior.coords.xy

        if x[0] == x[len(x)-1] and y[0] == y[len(y)-1]:
            x = x[:-1]
            y = y[:-1]

        return Polygon(x, y)

    def support_func(self, dir):
        """evaluate the support function of the polygon in the given direction"""

        v = np.transpose(self.vertices)
        return np.max(np.dot(dir, v))

    def contains(self, p):
        """check if the polygon contains a point"""

        pgon = geometry.Polygon([*zip(self.vertices[:, 0], self.vertices[:, 1])])
        return pgon.contains(geometry.Point(p[0], p[1]))

    def plot(self, color):
        """plot the polygon"""

        v = self.vertices
        v = np.concatenate((v, v[[0], :]), axis=0)
        plt.plot(v[:, 0], v[:, 1], color)

    def interval(self):
        """enclose the polygon by an interval"""

        x_min = np.min(self.vertices[:, 0])
        x_max = np.max(self.vertices[:, 0])
        y_min = np.min(self.vertices[:, 1])
        y_max = np.max(self.vertices[:, 1])

        return x_min, x_max, y_min, y_max

    def polytope(self):
        """convert the (convex) polygon to a polytope"""

        v = self.vertices
        v = np.concatenate((v, v[[0], :]), axis=0)
        v = np.transpose(v)

        C = np.zeros((v.shape[1]-1, 2))
        D = np.zeros((v.shape[1]-1, 1))

        for i in range(v.shape[1]-1):
            dir = v[:, i + 1] - v[:, i]
            c = np.array([[dir[1], -dir[0]]])
            d = np.dot(c, v[:, i])

            if np.max(np.dot(c, v) - d) > 0:
                C[i, :] = -c
                D[i] = -d
            else:
                C[i, :] = c
                D[i] = d

        return Polytope(C, D)

    def area(self):
        """compute the area of the polygon"""

        pgon = geometry.Polygon([*zip(self.vertices[:, 0], self.vertices[:, 1])])
        return pgon.area

    def is_valid(self):
        """check if the polygon is valid"""

        pgon = geometry.Polygon([*zip(self.vertices[:, 0], self.vertices[:, 1])])
        return pgon.is_valid

    def simplify(self):

        pgon = geometry.Polygon([*zip(self.vertices[:, 0], self.vertices[:, 1])])
        d = 0.00001  # distance
        cf = 1.3  # cofactor
        pgon = pgon.buffer(-d).buffer(d * cf).intersection(pgon).simplify(d)
        x, y = pgon.exterior.coords.xy

        return Polygon(x, y)

    def largest_convex_subset(self):
        """compute the largest convex subset of the polygon"""

        v = np.around(self.vertices, decimals=8)
        poly = Polygon(v[:, 0], v[:, 1])

        while True:

            # determine all non-convex line segments
            v = poly.vertices
            v = np.concatenate((v, v[0:2, :]), axis=0)
            v = v.transpose()

            ind = []
            convex = True

            for i in range(v.shape[1] - 2):

                d1 = v[:, i + 1] - v[:, i]
                d2 = v[:, i + 2] - v[:, i + 1]

                if d1[1] * d2[0] - d1[0] * d2[1] < -10**-5:
                    convex = False
                    ind.append(i+1)
                elif convex is False:
                    break

            if convex:
                break

            # determine best tangent line for each non-convex line segment
            area = -np.Inf
            ind.append(ind[len(ind)-1]+1)

            for i in ind:
                c = v[:, i] - v[:, i - 1]
                c = np.array([[c[1], -c[0]]])
                d = np.dot(c, v[:, i])
                poly_ = poly.intersection_halfspace(c, d)
                area_ = poly_.area()
                if poly_.is_valid() and area_ > area:
                    P = poly_
                    area = area_
                poly_ = poly.intersection_halfspace(-c, -d)
                area_ = poly_.area()
                if poly_.is_valid() and area_ > area:
                    P = poly_
                    area = area_

            poly = deepcopy(P.simplify())

        return poly

    def intersection_halfspace(self, c, d):
        """intersect the polygon with a halfspace {x | c*x <= d}"""

        v = self.vertices
        v = np.concatenate((v, v[[0], :]), axis=0)
        v = np.transpose(v)
        tmp = np.dot(c, v) - d < 10**-5
        tmp = tmp[0]

        v_ = []

        for i in range(len(tmp)-1):
            if tmp[i]:
                v_.append(v[:, i])
            if not tmp[i] == tmp[i+1]:
                p1 = v[:, i]
                p2 = v[:, i+1]
                dir = p2 - p1
                a = (d - np.dot(c, p1)) / np.dot(c, dir)
                v_.append(p1 + dir * a)

        v = np.asarray(v_)

        return Polygon(v[:, 0], v[:, 1])

if __name__ == '__main__':

    """test = loadmat('debut.mat')

    v = test['v']
    pgon_tmp = Polygon(v[:, 0], v[:, 1])
    test = pgon_tmp.largest_convex_subset()"""

    v = np.array([[3.71004472,  0.60536893],
                  [5.77968435,  0.67320848],
                  [5.77968435,  0.36776389],
                  [4.57615747,  0.24738788],
                  [4.74541561, -0.31059272],
                  [5.77968435, -0.3845218],
                  [5.77968435, -1.37895175],
                  [3.71004472, -1.37895175]])

    pgon = Polygon(v[:, 0], v[:, 1])
    test = pgon.largest_convex_subset()

    pgon.plot('r')
    test.plot('b')
    plt.show()

    lenpoly = 100
    polygon = [[np.sin(x) + 0.5, np.cos(x) + 0.5] for x in np.linspace(0, 2 * np.pi, lenpoly)[:-1]]

    x = np.array([0, 1, 1, 2, 0.5])
    y = np.array([0, 1, 0.5, 0, 0.2])
    poly = Polygon(x, y)

    poly_ = poly.intersection_halfspace(np.array([[-1, 0]]), -0.75)

    test = poly.largest_convex_subset()
    poly.plot('r')
    test.plot('g')
    plt.show()

    test = 1