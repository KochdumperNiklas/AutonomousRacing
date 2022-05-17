import numpy as np
import matplotlib.pyplot as plt

class Line:

    def __init__(self, p1, p2):
        self.p1 = np.resize(p1, (len(p1), 1))
        self.p2 = np.resize(p2, (len(p2), 1))
        self.len = np.linalg.norm(self.p2 - self.p1)
        self.d = (self.p2 - self.p1)/self.len

    def __mul__(self, M):
        return Line(np.dot(M, self.p1), np.dot(M, self.p2))

    def __add__(self, p):
        return Line(self.p1 + p, self.p2 + p)

    def center(self):
        return 0.5*(self.p1 + self.p2)

    def plot(self, color):
        plt.plot(np.array([self.p1[0], self.p2[0]]), np.array([self.p1[1], self.p2[1]]), color)

    def intersects(self, line):
        tmp = np.linalg.solve(np.concatenate((self.d, line.d), axis=1), line.center() - self.center())
        if abs(tmp[0]) < self.len/2 and abs(tmp[1]) < line.len/2:
            return True, self.center() + self.d * tmp[0]
        else:
            return False, None
