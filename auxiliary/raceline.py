import numpy as np
import math
from copy import deepcopy
from os.path import exists
from os.path import abspath
from os.path import dirname

def get_reference_trajectory(raceline, x, y, theta, v, N, dt, ind_prev):
    """extract the reference trajectory piece for the current state"""

    # initialization
    ref_traj = np.zeros((4, N + 1))
    dist_delta = np.sqrt(np.sum((raceline[[0, 1], 1] - raceline[[0, 1], 0]) ** 2, axis=0))
    dist = 0

    # get closest raceline point for the current state
    ind = closest_raceline_point(raceline, np.array([[x], [y]]), v, dt, ind_prev)

    # loop over all reference trajectory points
    for i in range(N + 1):

        # travelled distance based on current velocity
        dist += abs(v) * dt
        ind_delta = int(round(dist / dist_delta))
        ind_new = np.mod(ind + ind_delta, raceline.shape[1])

        # store reference trajectory point
        ref_traj[:, i] = raceline[:, ind_new]

        # consider heading change from 2pi -> 0 and 0 -> 2pi to guarantee that all headings are the same
        if raceline[3, ind_new] - theta > 5:
            ref_traj[3, i] = abs(raceline[3, ind_new] - 2 * math.pi)
        elif raceline[3, ind_new] - theta < -5:
            ref_traj[3, i] = abs(raceline[3, ind_new] + 2 * math.pi)

    return ref_traj, ind

def get_ref_traj_piece(raceline, x, y, theta, v, N, dt, ind_prev):
    """extract the relevent piece of the reference trajectory"""

    # initialization
    dist_delta = np.sqrt(np.sum((raceline[[0, 1], 1] - raceline[[0, 1], 0]) ** 2, axis=0))
    ind_delta = int(round(N * dt * abs(v) / dist_delta))

    # get closest raceline point for the current state
    ind = closest_raceline_point(raceline, np.array([[x], [y]]), v, dt, ind_prev)

    # extract relevant reference trajectory piece
    ref_traj = deepcopy(raceline[:, ind:ind + ind_delta])

    # consider heading change from 2pi -> 0 and 0 -> 2pi to guarantee that all headings are the same
    for i in range(ref_traj.shape[1]):
        if ref_traj[3, i] - theta > 5:
            ref_traj[3, i] = abs(ref_traj[3, i] - 2 * math.pi)
        elif ref_traj[3, i] - theta < -5:
            ref_traj[3, i] = abs(ref_traj[3, i] + 2 * math.pi)

    return ref_traj, ind

def closest_raceline_point(raceline, x, v, dt, ind_prev):
    """find the point on the raceline that is closest to the current state"""

    # determine search range
    dist_delta = np.sqrt(np.sum((raceline[[0, 1], 1] - raceline[[0, 1], 0]) ** 2, axis=0))
    ind_diff = np.ceil(v*dt/dist_delta) + 10

    ind_range = np.mod(np.arange(ind_prev, ind_prev + ind_diff), raceline.shape[1]).astype(int)

    # compute closest point
    ind = np.argmin(np.sum((raceline[0:2, ind_range]-x)**2, axis=0))

    return ind_range[ind]

def load_raceline(path):
    """load the optimal raceline from the corresponding file"""

    path = dirname(dirname(abspath(__file__))) + '/' + path

    if not exists(path):
        raise Exception('No file raceline.csv found for this racetrack!')

    tmp = np.loadtxt(path, delimiter=';', skiprows=3)
    return np.transpose(tmp[:, [1, 2, 5, 3]])
