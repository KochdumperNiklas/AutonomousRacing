import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from copy import deepcopy
import pypoman

class Polytope:
# class representing a polytope in halfspace representation

    def __init__(self, c, d):
    # class constructor

        self.c = c                          # matrix C for the inequality constraint C*x <= d
        self.d = d                          # constant offset for the inequality constraint C*x <= d

    def __sub__(self, set):
    # Minkowski difference of the polytope with another set

        d = deepcopy(self.d)

        # loop over all halfspaces
        for i in range(self.c.shape[0]):
            tmp = set.support_func(self.c[i, :])
            d[i] = d[i] - tmp

        return Polytope(self.c, d)

    def intersects(self, p):
    # check if the polytope intersects a point cloud p
        tmp = np.dot(self.c, p) - np.dot(np.resize(self.d, (len(self.d), 1)), np.ones((1, p.shape[1])))
        return not np.all(np.max(tmp, 0) > 0)

    def shift(self, p):
    # shift the polytope by a vector p
        self.d = self.d + np.dot(self.c, p)

    def rotate(self, phi):
    # rotate the polytope by the angle phi
        self.c = np.dot(self.c, np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]]))

    def vertices(self):
    # compute the vertices of a polytope
        return pypoman.compute_polytope_vertices(self.c, self.d)

    def plot(self, color):
    # plot the polytope

        v = self.vertices()
        v.append(v[0])
        v_ = np.resize(v[0], (2, 1))
        for i in range(1, len(v)):
            v_ = np.concatenate((v_, np.resize(v[i], (2, 1))), axis=1)
        hull = ConvexHull(np.transpose(v_))
        for simplex in hull.simplices:
            plt.plot(v_[0, simplex], v_[1, simplex], color)
