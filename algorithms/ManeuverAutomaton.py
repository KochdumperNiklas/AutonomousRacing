import time
import yaml
import gym
import pypoman
import operator
import matplotlib.pyplot as plt
import numpy as np
from os.path import exists
from copy import deepcopy
from auxiliary.process_lidar_data import process_lidar_data
from auxiliary.Polytope import Polytope
from auxiliary.vehicle_model import simulate
from auxiliary.raceline import load_raceline
from auxiliary.raceline import get_ref_traj_piece


""" --------------------------------------------------------------------------------------------------------------------
Motion Planner
-------------------------------------------------------------------------------------------------------------------- """

class ManeuverAutomaton:
    """class representing a maneuver automaton motion planner"""

    def __init__(self, params, settings):
        """class constructor"""

        # store algorithm settings
        settings['Q'] = np.diag(settings['Q'])
        self.settings = settings

        # load optimal raceline
        if settings['RACELINE']:
            self.raceline = load_raceline(settings['path_raceline'])
            self.ind_prev = 0

        # initial control input
        self.u_prev = np.array([[0], [0]])

        # wheelbase and width/length of the car
        self.params = params
        self.WB = params['lf'] + params['lr']
        self.width = params['width']
        self.length = params['length']

        # construct motion primitives
        motion_primitives, conn_mat, indices = self.construct_automaton()

        # store motion primitives
        self.motion_primitives = motion_primitives      # list of motion primitives
        self.conn_mat = conn_mat                        # connectivity matrix (which motion primitives can be connected)
        self.indices = indices                          # list of indices mapping init. velocities to motion primitives

        # initialize counter for re-planning
        self.control_lim = np.ceil(1 / (settings['freq'] * 0.01)).astype(int)
        self.control_count = self.control_lim

    def plan(self, x, y, theta, v, scans):
        """plan a trajectory"""

        # check if control frequency is reached
        self.control_count += 1

        if self.control_count <= self.control_lim:
            return self.u_prev
        else:
            self.control_count = 0

        # transform lidar data into point cloud
        points = process_lidar_data(scans)

        # get reference trajectory
        ref_traj = None

        if self.settings['RACELINE']:
            v_ = max(v, 0.1)
            ref_traj, ind = get_ref_traj_piece(self.raceline, x, y, theta, v_,
                                               self.settings['N']-1, self.settings['DT'], self.ind_prev)
            rot_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
            ref_traj[0:2, :] = ref_traj[0:2] - np.array([[x], [y]])
            ref_traj[0:2, :] = np.dot(rot_mat, ref_traj[0:2, :])
            ref_traj[3, :] = ref_traj[3, :] - theta
            self.ind_prev = ind

        # initialize queue
        queue = []
        ind = np.argmin(np.abs(np.asarray(self.settings['vel_init']) - v))

        for i in self.indices[ind]:
            mp = self.motion_primitives[i]
            if not mp.occupancy_set.intersects(points):
                cost = self.cost_function(mp.xEnd, ref_traj, points)
                queue.append(Node(mp.xEnd, [i], cost))

        # loop until the queue is empty
        cost = np.Infinity
        best = []

        while queue:

            # pop element from queue
            queue.sort(key=operator.attrgetter('cost'))
            node = deepcopy(queue[0])
            queue = queue[1:len(queue)]

            # check if planning horizon is reached
            if len(node.ind) == self.settings['N']:

                if node.cost < cost:
                    best = deepcopy(node)
                    break

            else:

                # add child nodes to queue
                index = node.ind[len(node.ind)-1]
                ind = np.nonzero(self.conn_mat[index, :])

                for i in ind[0]:

                    # update occupancy set
                    mp = deepcopy(self.motion_primitives[i])
                    occ_set = deepcopy(mp.occupancy_set)
                    occ_set.transform(node.x)

                    # collision checking
                    if not occ_set.intersects(points):

                        mp = deepcopy(self.motion_primitives[i])
                        x_ = deepcopy(transform_state(mp.xEnd, deepcopy(node.x)))
                        cost_ = node.cost + self.cost_function(x_, ref_traj, points) - 100*len(node.ind)
                        ind_ = deepcopy(node.ind)
                        ind_.append(i)
                        queue.append(Node(x_, ind_, cost_))

        # return control input for the best trajectory
        if not best:
            u = self.u_prev
            print('Failed to find a feasible solution!')
        else:
            u = self.motion_primitives[best.ind[0]].u
            self.u_prev = u

        # visualize the planned trajectory
        if self.settings['VISUALIZE']:
            self.visualization(x, y, theta, points, best)

        return u

    def construct_automaton(self):
        """construct the maneuver automaton"""

        # initialization
        v_max = max(self.settings['vel_init'])
        v_min = min(self.settings['vel_init'])

        list_MP = []
        list_v = []
        cnt = 0
        t = np.linspace(0, self.settings['DT'], self.settings['N_STEPS'])

        # loop over all initial velocities
        for v_init in self.settings['vel_init']:

            # select suitable desired velocities that can be reached from the current velocity
            ind = []
            vel = v_init + np.asarray(self.settings['vel'])
            vel = vel[vel > v_min]
            vel = vel[vel <= v_max]

            # loop over all steering angles
            for steer in self.settings['steer']:

                # loop over all desired velocities
                for v in vel:

                    # simulate the car
                    u = np.array([v, steer])
                    x0 = np.array([0, 0, 0, v, 0, 0, 0])
                    x = simulate(x0, u, t, self.params)

                    # construct the motion primitive
                    length = self.length * self.settings['scale_length']
                    width = self.width * self.settings['scale_width']
                    list_MP.append(MotionPrimitive(x, u, length, width, self.settings['UNITE']))
                    ind.append(cnt)
                    cnt = cnt + 1

            list_v.append(np.asarray(ind))

        # construct connectivity matrix
        conn_mat = np.zeros((len(list_MP), len(list_MP)))

        for i in range(1, len(list_MP)):
            ind = np.argmin(np.abs(self.settings['vel_init'] - list_MP[i].xEnd[3]))
            conn_mat[i, list_v[ind]] = np.ones((1, len(list_v[ind])))

        return list_MP, conn_mat, list_v

    def visualization(self, x, y, theta, lidar_data, best):
        """visualize the planned trajectory"""

        if best:
            plt.cla()
            plt.plot(lidar_data[0, :], lidar_data[1, :], '.r', label='lidar measurements')
            if self.settings['RACELINE']:
                rl = self.raceline[0:2, :] - np.array([[x], [y]])
                rl = np.dot(np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]), rl)
                plt.plot(rl[0, :], rl[1, :], 'g', label='optimal raceline')
            plot_trajectory(self.motion_primitives, best.ind, 'b')
            plt.axis('equal')
            x_min = np.min(lidar_data[0, :])-2
            x_max = np.max(lidar_data[0, :])+2
            y_min = np.min(lidar_data[1, :])-2
            y_max = np.max(lidar_data[1, :])+2
            plt.plot([x_max + 1, x_max + 1], [y_max + 1, y_max + 1], 'b', label='occupancy set planned trajectory')
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.legend(loc='upper right')
            plt.pause(0.1)

    def cost_function(self, x, ref_traj, lidar_data):
        """cost function"""

        if self.settings['RACELINE']:
            tmp = ref_traj - np.resize(x[[0, 1, 3, 4]], (4, 1))
            ind = np.argmin(np.sum(tmp[0:2, :]**2, axis=0))
            cost = np.dot(np.transpose(tmp[:, ind]), np.dot(self.settings['Q'], tmp[:, ind]))
        else:
            tmp = (lidar_data - np.dot(np.resize(x[0:2], (2, 1)), np.ones((1, lidar_data.shape[1])))) ** 2
            cost = -min(np.sum(tmp, 0))

        return cost


""" --------------------------------------------------------------------------------------------------------------------
Auxiliary Functions 
-------------------------------------------------------------------------------------------------------------------- """

def transform_state(x, x_new):
    """shift the state x of the car to the new position x_new (possible due to translation and rotation invariance)"""

    # get orientation and x- and y-position
    phi = x_new[4]
    x_temp = x[0:2]

    # compute the rotation matrix from the orientation
    rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

    # update state
    x_new[0:2] = x_new[0:2] + np.dot(rot_mat, x_temp)
    x_new[4] = x_new[4] + x[4]

    return x_new

def plot_trajectory(list_mp, ind, color):
    """plot a planed trajectory specified by the indices ind of the corresponding motion primitives"""

    x = np.zeros((7, ))

    # loop over all motion primitives that are part of the planned trajectory
    for i in ind:

        # translate the occupancy set to the current state x
        occ_set = deepcopy(list_mp[i].occupancy_set)
        occ_set.transform(x)

        # plot the occupancy set
        occ_set.plot(color)

        # update state of the car
        x = transform_state(list_mp[i].xEnd, x)


""" --------------------------------------------------------------------------------------------------------------------
Auxiliary Classes 
-------------------------------------------------------------------------------------------------------------------- """

class MotionPrimitive:
# class representing a single motion primitive

    def __init__(self, x, u, length, width, unite):
    # class constructor

        # compute occupancy set
        occ_set = self.construct_occupancy_set(x, length, width, unite)

        # object properties
        self.x = x                              # trajectory of the cars reference point
        self.u = u                              # control inputs
        self.xStart = x[0, :]                   # initial state of the trajectory
        self.xEnd = x[x.shape[0]-1, :]          # final state of the trajectory
        self.occupancy_set = occ_set            # occupancy set (space occupied by the car)

    def construct_occupancy_set(self, x, length, width, unite):
    # compute the space occupied by the car (represented as a union of polytopes)

        # vertices of the car
        x_min = -length/2
        x_max = length/2

        vert = np.array([[x_max, x_max, x_min, x_min], [width/2, -width/2, -width/2, width/2]])

        # group trajectory points together into batches
        batches = np.arange(0, x.shape[0], unite)
        batches = np.concatenate((batches, np.array([x.shape[0]])))

        # loop over all batches to create the corresponding polytope
        polytopes = []

        for i in range(0, len(batches)-1):

            points = (x[batches[i], :], x[batches[i+1]-1, :], x[int(np.floor(0.5*(batches[i]+batches[i+1]-1))), :])

            # create template directions
            C = np.zeros((12, 2))
            d = -np.Infinity*np.ones((12, 1))
            cnt = 0

            for p in points:
                C[cnt, :] = np.array([-np.sin(p[4]), np.cos(p[4])])
                C[cnt+1, :] = np.array([np.cos(p[4]), np.sin(p[4])])
                C[cnt+2, :] = np.array([np.sin(p[4]), -np.cos(p[4])])
                C[cnt+3, :] = np.array([-np.cos(p[4]), -np.sin(p[4])])
                cnt = cnt + 4

            # determine polytope offset along the template directions
            for j in range(batches[i], batches[i+1]):
                phi = x[j][4]
                rot_mat = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
                vert_ = np.dot(rot_mat, vert) + np.resize(x[j][0:2], (2, 1))
                d_ = np.max(np.dot(C, vert_), 1)
                d = np.max(np.concatenate((np.resize(d, (12, 1)), np.resize(d_, (12, 1))), axis=1), 1)

            # add resulting polytope to the list
            polytopes.append(Polytope(C, d))

        return OccupancySet(polytopes)


class OccupancySet:
# class represening the occupancy set

    def __init__(self, polytopes):
    # class constructor

        self.polytopes = polytopes                  # list storing the polytopes that represent the occupancy set

    def intersects(self, p):
    # check if the occupancy set intersect a point cloud p
        return np.any([poly.intersects(p) for poly in self.polytopes])

    def transform(self, x):
    # move the occupancy set to a different state x

        p = x[0:2]              # x- and y-position
        phi = x[4]              # orientation

        # loop over all polytopes
        for i in range(0, len(self.polytopes)):
            poly = self.polytopes[i]
            poly.rotate(phi)
            poly.shift(p)
            self.polytopes[i] = poly

    def plot(self, color):
    # plot the occupancy set

        for p in self.polytopes:
            try:
                p.plot(color)
            except:
                test = 1

class Node:
# class representing a node of the search tree

    def __init__(self, x, ind, cost):
        self.x = x                          # current state of the car
        self.ind = ind                      # indices of the motion primitives
        self.cost = cost                    # cost of the current node


