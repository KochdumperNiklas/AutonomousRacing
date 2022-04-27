import numpy as np
from copy import deepcopy
from auxiliary.Polygon import Polygon

def free_space(points):
    """compute a polygon that represents the free space from the lidar data"""

    # approximate point-cloud with line-segments
    tmp = linesegment_refinement(points)

    # convert line-segments to polygon
    vert = np.zeros((len(tmp) + 1, 2))

    for i in range(0, len(tmp)):
        vert[i, :] = tmp[i][:, 0]

    vert[len(tmp), :] = tmp[len(tmp) - 1][:, tmp[len(tmp) - 1].shape[1] - 1]
    vert = np.transpose(vert)

    # contract the polygon so that all points are outside
    vert = np.concatenate((vert[:, [vert.shape[1]-1]], vert, vert[:, [0]]), axis=1)

    for i in range(1, vert.shape[1]-2):

        # convert current line-segment to halfspace
        dir = vert[:, i + 1] - vert[:, i]
        c = np.array([[-dir[1], dir[0]]])
        d = np.max(np.dot(c, tmp[i-1]))

        # intersect halfspace with the two neighbouring line-segments
        dir = vert[:, i] - vert[:, i - 1]
        a = (d - np.dot(c, vert[:, i - 1])) / np.dot(c, dir)
        p1 = vert[:, i - 1] + dir * a

        dir = vert[:, i + 2] - vert[:, i + 1]
        a = (d - np.dot(c, vert[:, i + 1])) / np.dot(c, dir)
        p2 = vert[:, i + 1] + dir * a

        vert[:, i] = p1
        vert[:, i + 1] = p2

    return Polygon(vert[0, 1:-1], vert[1, 1:-1])


def linesegment_refinement(points):
    """refine line segments to improve a line-segment fit for a given point cloud"""

    d1 = points[:, points.shape[1]-1] - points[:, 0]
    d1 = d1 / np.linalg.norm(d1)
    d2 = np.resize(np.array([d1[1], -d1[0]]), (1, 2))

    tmp = np.dot(d2, points) - np.dot(d2, points[:, 0])
    ind = np.argmax(abs(tmp))

    if abs(tmp[0, ind]) > 0.2:
        points1 = linesegment_refinement(deepcopy(points[:, 0:ind+1]))
        points2 = linesegment_refinement(deepcopy(points[:, ind:points.shape[1]]))
        return points1 + points2
    else:
        return [points]
