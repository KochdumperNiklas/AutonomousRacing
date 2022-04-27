import numpy as np
import matplotlib.path as mpltPath
from shapely import geometry
import matplotlib.pyplot as plt
from copy import deepcopy
from auxiliary.Polytope import Polytope

class Polygon:
    """class represnting a polygon"""

    def __init__(self, x, y):
        """class constructor"""

        x = np.resize(x, (len(x), 1))
        y = np.resize(y, (len(y), 1))
        tmp = np.concatenate((x, y), axis=1)

        self.set = mpltPath.Path(tmp.tolist())

    def __and__(self, set):
        """intersection of two polygons"""

        # compute intersection
        pgon1 = geometry.Polygon([*zip(self.set.vertices[:, 0], self.set.vertices[:, 1])])
        pgon2 = geometry.Polygon([*zip(set.set.vertices[:, 0], set.set.vertices[:, 1])])

        pgon = pgon1.intersection(pgon2)

        x, y = pgon.exterior.coords.xy

        # remove redundant vertices
        if x[0] == x[len(x)-1] and y[0] == y[len(y)-1]:
            x = x[:-1]
            y = y[:-1]

        return Polygon(x, y)

    def support_func(self, dir):
        """evaluate the support function of the polygon in the given direction"""

        v = np.transpose(self.set.vertices)
        return np.max(np.dot(dir, v))

    def plot(self, color):
        """plot the polygon"""

        v = self.set.vertices
        v = np.concatenate((v, v[[0], :]), axis=0)
        plt.plot(v[:, 0], v[:, 1], color)

    def interval(self):
        """enclose the polygon by an interval"""

        x_min = np.min(self.set.vertices[:, 0])
        x_max = np.max(self.set.vertices[:, 0])
        y_min = np.min(self.set.vertices[:, 1])
        y_max = np.max(self.set.vertices[:, 1])

        return x_min, x_max, y_min, y_max

    def polytope(self):
        """convert the (convex) polygon to a polytope"""

        v = self.set.vertices
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

        pgon = geometry.Polygon([*zip(self.set.vertices[:, 0], self.set.vertices[:, 1])])
        return pgon.area

    def largest_convex_subset(self):
        """compute the largest convex subset of the polygon"""

        poly = deepcopy(self)

        while True:

            # determine all non-convex line segments
            v = poly.set.vertices
            v = np.concatenate((v, v[0:2, :]), axis=0)
            v = v.transpose()

            ind = []
            convex = True

            for i in range(v.shape[1] - 2):

                d1 = v[:, i + 1] - v[:, i]
                d2 = v[:, i + 2] - v[:, i + 1]

                if d1[1] * d2[0] - d1[0] * d2[1] < 0:
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
                if area_ > area:
                    P = poly_
                    area = area_
                poly_ = poly.intersection_halfspace(-c, -d)
                area_ = poly_.area()
                if area_ > area:
                    P = poly_
                    area = area_

            poly = deepcopy(P)

        return poly

    def intersection_halfspace(self, c, d):
        """intersect the polygon with a halfspace {x | c*x <= d}"""

        v = self.set.vertices
        v = np.concatenate((v, v[[0], :]), axis=0)
        v = np.transpose(v)
        tmp = np.dot(c, v) - d < 0
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

    v = np.array([[1.39738069, 0.8082182 ],
                  [2.88387841, 0.47403599],
                  [4.00238069, 0.64136782],
                  [4.00238069, -1.11533437],
                  [1.39738069, -1.11533437]])

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