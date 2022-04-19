import time
import yaml
import gym
import pypoman
import operator
import matplotlib.pyplot as plt
import numpy as np
from argparse import Namespace
from scipy.integrate import odeint
from scipy.spatial import ConvexHull
from copy import deepcopy


""" --------------------------------------------------------------------------------------------------------------------
Vehicle Model 
-------------------------------------------------------------------------------------------------------------------- """

def simulate(x0, u, t, params):
    """simulate the car"""

    sol = odeint(dynamic_function, x0, t, args=(u, params))
    return sol

def dynamic_function(x,t,u,params):
    """differential equation for the car-model"""

    # apply PID controller to determine acceleration and steering velocity from desired velocity and steering angle
    accl, sv = pid(u[0], u[1], x[3], x[2], params['sv_max'], params['a_max'], params['v_max'], params['v_min'])

    # get right-hand side of the differential equation
    f = vehicle_dynamics_st(x, np.array([sv, accl]), params['mu'], params['C_Sf'], params['C_Sr'], params['lf'],
                            params['lr'], params['h'], params['m'], params['I'], params['s_min'], params['s_max'],
                            params['sv_min'], params['sv_max'], params['v_switch'], params['a_max'], params['v_min'],
                                                                                                       params['v_max'])
    return f

def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """constraints for acceleration"""

    # positive acceleration limit
    if vel > v_switch:
        pos_limit = a_max*v_switch/vel
    else:
        pos_limit = a_max

    # acceleration limit reached?
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl

def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """constraints for steering"""

    # constraint steering velocity
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity

def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch,
                        a_max, v_min, v_max):
    """kinematic single track vehicle model"""

    # wheelbase
    lwb = lf + lr

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1],
                  v_switch, a_max, v_min, v_max)])

    # system dynamics
    f = np.array([x[3]*np.cos(x[4]),
         x[3]*np.sin(x[4]),
         u[0],
         u[1],
         x[3]/lwb*np.tan(x[2])])#

    return f

def vehicle_dynamics_st(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch, a_max,
                        v_min, v_max):
    """single track vehicle model"""

    # gravity constant m/s^2
    g = 9.81

    # constraints
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max), accl_constraints(x[3], u_init[1],
                  v_switch, a_max, v_min, v_max)])

    # switch to kinematic model for small velocities
    if abs(x[3]) < 0.5:

        lwb = lf + lr       # wheelbase

        # system dynamics
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I, s_min, s_max, sv_min, sv_max, v_switch,
                                   a_max, v_min, v_max)
        f = np.hstack((f_ks, np.array([u[1]/lwb*np.tan(x[2])+x[3]/(lwb*np.cos(x[2])**2)*u[0],
        0])))

    else:
        f = np.array([x[3]*np.cos(x[6] + x[4]),
            x[3]*np.sin(x[6] + x[4]),
            u[0],
            u[1],
            x[5],
            -mu*m/(x[3]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-u[1]*h) + lr**2*C_Sr*(g*lf + u[1]*h))*x[5] \
                +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + u[1]*h) - lf*C_Sf*(g*lr - u[1]*h))*x[6] \
                +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - u[1]*h)*x[2],
            (mu/(x[3]**2*(lr+lf))*(C_Sr*(g*lf + u[1]*h)*lr - C_Sf*(g*lr - u[1]*h)*lf)-1)*x[5] \
                -mu/(x[3]*(lr+lf))*(C_Sr*(g*lf + u[1]*h) + C_Sf*(g*lr-u[1]*h))*x[6] \
                +mu/(x[3]*(lr+lf))*(C_Sf*(g*lr-u[1]*h))*x[2]])

    return f

def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """low-level PID controller for speed and steering"""

    # steering
    steer_diff = steer - current_steer
    if np.fabs(steer_diff) > 1e-4:
        sv = (steer_diff / np.fabs(steer_diff)) * max_sv
    else:
        sv = 0.0

    # acceleration
    vel_diff = speed - current_speed

    # currently forward
    if current_speed > 0.:
        if vel_diff > 0:
            # accelerate
            kp = 10.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # braking
            kp = 10.0 * max_a / (-min_v)
            accl = kp * vel_diff

    # currently backwards
    else:
        if vel_diff > 0:
            # braking
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            # accelerating
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl, sv


""" --------------------------------------------------------------------------------------------------------------------
Auxiliary Functions 
-------------------------------------------------------------------------------------------------------------------- """

def process_lidar_data(data):
    """convert the lidar measurement to a 2-dimenaional point cloud"""

    # parameter (defined by the lidar that is used)
    phi = -2.35619449615                                # initial angle
    delta_phi = 0.00436332309619                        # angle increment

    # loop over all lidar beams
    points = np.zeros((2, len(data)))

    for i in range(0, len(data)):
        points[0, i] = data[i] * np.cos(phi)
        points[1, i] = data[i] * np.sin(phi)
        phi = phi + delta_phi

    return points

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

    def __init__(self, x, u, length, width,  unite):
    # class constructor

        # compute occupancy set
        occ_set = self.construct_occupancy_set(x, u, length, width,  unite)

        # object properties
        self.x = x                              # trajectory of the cars reference point
        self.u = u                              # control inputs
        self.xStart = x[0, :]                   # initial state of the trajectory
        self.xEnd = x[x.shape[0]-1, :]          # final state of the trajectory
        self.occupancy_set = occ_set            # occupancy set (space occupied by the car)


    def construct_occupancy_set(self, x, u, length, width, unite):
    # compute the space occupied by the car (represented as a union of polytopes)

        # vertices of the car
        vert = np.array([[length, length, -length, -length], [width, -width, -width, width]])

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


class Polytope:
# class representing a polytope in halfspace representation

    def __init__(self, c, d):
    # class constructor

        self.c = c                          # matrix C for the inequality constraint C*x <= d
        self.d = d                          # constant offset for the inequality constraint C*x <= d

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


class Node:
# class representing a node of the search tree

    def __init__(self, x, ind, cost):
        self.x = x                          # current state of the car
        self.ind = ind                      # indices of the motion primitives
        self.cost = cost                    # cost of the current node


""" --------------------------------------------------------------------------------------------------------------------
Motion Planner
-------------------------------------------------------------------------------------------------------------------- """

class ManeuverAutomaton:
    """class representing a maneuver automaton motion planner"""

    def __init__(self, racetrack, params):
        """class constructor"""

        # load controller settings
        self.load_controller_settings(racetrack)

        # store car parameter (mass, width ,length, etc.)
        self.params = params

        # construct motion primitives
        motion_primitives, conn_mat, indices = self.construct_automaton()

        # store motion primitives
        self.motion_primitives = motion_primitives      # list of motion primitives
        self.conn_mat = conn_mat                        # connectivity matrix (which motion primitives can be connected)
        self.indices = indices                          # list of indices mapping init. velocities to motion primitives

    def construct_automaton(self):
        """construct the maneuver automaton"""

        # initialization
        v_max = max(self.vel_init)
        v_min = min(self.vel_init)

        list_MP = []
        list_v = []
        cnt = 0
        t = np.linspace(0, self.DT, self.N_STEPS)

        # loop over all initial velocities
        for v_init in self.vel_init:

            # select suitable desired velocities that can be reached from the current velocity
            ind = []
            vel = v_init + np.asarray(self.vel)
            vel = vel[vel > v_min]
            vel = vel[vel <= v_max]

            # loop over all steering angles
            for steer in self.steer:

                # loop over all desired velocities
                for v in vel:

                    # simulate the car
                    u = np.array([v, steer])
                    x0 = np.array([0, 0, 0, v, 0, 0, 0])
                    x = simulate(x0, u, t, self.params)

                    # construct the motion primitive
                    list_MP.append(MotionPrimitive(x, u, self.params['length'], self.params['width'], self.UNITE))
                    ind.append(cnt)
                    cnt = cnt + 1

            list_v.append(np.asarray(ind))

        # construct connectivity matrix
        conn_mat = np.zeros((len(list_MP), len(list_MP)))

        for i in range(1, len(list_MP)):
            ind = np.argmin(np.abs(self.vel_init - list_MP[i].xEnd[3]))
            conn_mat[i, list_v[ind]] = np.ones((1, len(list_v[ind])))

        return list_MP, conn_mat, list_v

    def load_controller_settings(self, racetrack):
        """load settings for the MPC controller"""

        # load controller settings from file
        path = 'racetracks/' + racetrack + '/settings_' + racetrack + '_ManeuverAutomaton.txt'

        with open(path) as f:
            lines = f.readlines()

        for l in lines:
            exec(l, globals(), globals())

        # motion planning parameter
        self.DT = DT                                        # time step size [s]
        self.N = N                                          # number of time steps for MPC prediction
        self.freq = freq                                    # planning frequency [Hz]

        # motion primitives
        self.vel_init = vel_init                            # initial velocities for motion primitives
        self.steer = steer                                  # steering inputs for motion primitives
        self.vel = vel                                      # velocity inputs for motion primitives

        # additional settings
        self.N_STEPS = N_STEPS                              # number of time steps for one motion primitive
        self.UNITE = UNITE                                  # number of time steps that are united for occupancy set
        self.LIDAR_STEP = LIDAR_STEP                        # number of lidar points that are skipped

    def plan(self, x, y, theta, v, scans):
        """plan a trajectory"""

        # transform lidar data into point cloud
        points = process_lidar_data(scans)

        # initialize queue
        queue = []
        ind = np.argmin(np.abs(np.asarray(self.vel_init) - v))

        for i in self.indices[ind]:
            mp = self.motion_primitives[i]
            if not mp.occupancy_set.intersects(points):
                # cost = -np.sqrt(mp.xEnd[0]**2 + mp.xEnd[1]**2)
                cost = -min(np.sum((points - np.dot(np.resize(mp.xEnd[0:2], (2, 1)), np.ones((1, 1080)))) ** 2, 0))
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
            if len(node.ind) == self.N:

                if node.cost < cost:
                    best = deepcopy(node)
                    cost = node.cost
                    return self.motion_primitives[best.ind[0]].u

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
                        cost_ = -min(np.sum((points - np.dot(np.resize(x_[0:2], (2, 1)), np.ones((1, 1080)))) ** 2, 0))
                        # cost_ = -np.sqrt(x_[0]**2 + x_[1]**2)
                        ind_ = deepcopy(node.ind)
                        ind_.append(i)
                        queue.append(Node(x_, ind_, cost_))

        # return control input for the best trajectory
        if not best:
            raise Exception('Failed to find a feasible solution!')

        return self.motion_primitives[best.ind[0]].u
