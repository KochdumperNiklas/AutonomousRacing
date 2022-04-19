import time
import yaml
import gym
import numpy as np
from argparse import Namespace
import math
from numba import njit
import cvxpy
import matplotlib.pyplot as plt
import pickle
import copy

""" --------------------------------------------------------------------------------------------------------------------
Controller
-------------------------------------------------------------------------------------------------------------------- """

class MPC_Controller:
    """class representing an MPC controller that tracks the optimal raceline"""

    def __init__(self, path):
        """object constructor"""

        self.raceline = self.load_raceline(path)                        # load optimal raceline from file
        self.controller_settings()                                      # load controller settings

        # initialize previous control inputs and raceline index
        self.u_prev = np.zeros((2, self.N))
        self.ind_prev = 0

    def controller_settings(self):
        """settings for the MPC controller"""

        # weighting matrices for MPC
        self.R = np.diag([0.01, 100.0])                                 # input cost matrix[accel, steer]
        self.Rd = np.diag([0.01, 100.0])                                # input difference cost matrix[accel, steer]
        self.Q = np.diag([13.5, 13.5, 5.5, 13.0])                       # state cost matrix [x, y, v, yaw]
        self.Qf = np.diag([13.5, 13.5, 5.5, 13.0])                      # final state cost matrix [x, y, v, yaw]

        # motion planning parameter
        self.DT = 0.10                                                  # time step size [s]
        self.N = 10                                                     # number of time steps for MPC prediction
        self.freq = 1/self.DT                                           # planning frequency [Hz]

        # input and state constraints
        self.MAX_STEER = np.deg2rad(24.0)                               # maximum steering angle [rad]
        self.MAX_DSTEER = np.deg2rad(180.0)                             # maximum steering speed [rad/s]
        self.MAX_SPEED = 6                                              # maximum speed [m/s]
        self.MIN_SPEED = 0                                              # minimum backward speed [m/s]
        self.MAX_ACCEL = 2.5                                            # maximum acceleration [m/s**2]

        # Vehicle parameters
        self.LENGTH = 0.58                                              # length of the vehicle [m]
        self.WIDTH = 0.31                                               # width of the vehicle [m]
        self.WB = 0.33                                                  # length of the wheelbase [m]

    def plan(self, x, y, theta, v):
        """plan a trajectory using MPC"""

        v = np.max((v, 0.1))

        # get reference trajectory
        ref_traj = self.get_reference_trajectory(x, y, theta, v)

        # predict expected trajectory based on the previous control inputs (for linearization points)
        pred_traj = self.get_predicted_trajectory(x, y, theta, v)

        # plan new trajectory using MPC
        x0 = np.array([x, y, v, theta])
        self.u_prev = self.mpc_optimization(x0, ref_traj, pred_traj)

        return v + self.u_prev[0, 0] * self.DT, self.u_prev[1, 0]

    def mpc_optimization(self, x0, ref_traj, pred_traj):
        """solve optimal control problem for MPC"""

        # initialization
        x = cvxpy.Variable((4, self.N + 1))
        u = cvxpy.Variable((2, self.N))
        objective = 0.0
        constraints = []

        # objective function for optimal control problem
        for i in range(self.N):

            objective += cvxpy.quad_form(u[:, i], self.R)

            if i != 0:
                objective += cvxpy.quad_form(ref_traj[:, i] - x[:, i], self.Q)

            A, B, C = self.linearized_dynamic_function(pred_traj[2, i], pred_traj[3, i], 0)
            constraints += [x[:, i + 1] == A @ x[:, i] + B @ u[:, i] + C]

            if i < (self.N - 1):
                objective += cvxpy.quad_form(u[:, i + 1] - u[:, i], self.Rd)
                constraints += [cvxpy.abs(u[1, i + 1] - u[1, i]) <= self.MAX_DSTEER * self.DT]

        objective += cvxpy.quad_form(ref_traj[:, self.N] - x[:, self.N], self.Qf)

        # constraints for the optimal control problem
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.MAX_SPEED]
        constraints += [x[2, :] >= self.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_STEER]

        # solve the optimal control problem
        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            u = u.value
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u

    def get_reference_trajectory(self, x, y, theta, v):
        """extract the reference trajectory piece for the current state"""

        # initialization
        ref_traj = np.zeros((4, self.N+1))
        dist_delta = np.sqrt(np.sum((self.raceline[[0, 1], 1] - self.raceline[[0, 1], 0])**2, axis=0))
        dist = 0

        # get closest raceline point for the current state
        ind = self.closest_raceline_point(np.array([[x], [y]]), v)

        # loop over all reference trajectory points
        for i in range(self.N + 1):

            # travelled distance based on current velocity
            dist += abs(v) * self.DT
            ind_delta = int(round(dist / dist_delta))
            ind_new = np.mod(ind + ind_delta, self.raceline.shape[1])

            # store reference trajectory point
            ref_traj[:, i] = self.raceline[:, ind_new]

            # consider heading change from 2pi -> 0 and 0 -> 2pi to guarantee that all headings are the same
            if self.raceline[3, ind_new] - theta > 5:
                ref_traj[3, i] = abs(self.raceline[3, ind_new] - 2 * math.pi)
            elif self.raceline[3, ind_new] - theta < -5:
                ref_traj[3, i] = abs(self.raceline[3, ind_new] + 2 * math.pi)

        return ref_traj

    def get_predicted_trajectory(self, x, y, theta, v):
        """predict trajectory based on the previous control inputs"""

        # initialization
        state = np.array([x, y, v, theta])
        pred_traj = np.zeros((4, self.N+1))
        pred_traj[:, 0] = state

        # loop over all time steps
        for i in range(self.u_prev.shape[1]):
            pred_traj[:, i+1] = pred_traj[:, i] + self.dynamic_function(pred_traj[:, i], self.u_prev[:, i]) * self.DT
            pred_traj[2, i+1] = np.max((np.min((pred_traj[2, i+1], self.MAX_SPEED)), self.MIN_SPEED))

        return pred_traj

    def dynamic_function(self, x, u):
        """differential equation for the kinematic single track model"""

        u[1] = np.max((np.min((u[1], self.MAX_STEER)), -self.MAX_STEER))

        return np.array([x[2] * math.cos(x[3]),
                         x[2] * math.sin(x[3]),
                         u[0],
                         x[2] / self.WB * math.tan(u[1])])

    def linearized_dynamic_function(self, v, theta, steer):
        """linear time-discrete version of the kinematic single track model"""

        # system matrix A
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.DT * math.cos(theta)
        A[0, 3] = - self.DT * v * math.sin(theta)
        A[1, 2] = self.DT * math.sin(theta)
        A[1, 3] = self.DT * v * math.cos(theta)
        A[3, 2] = self.DT * math.tan(steer) / self.WB

        # input matrix B
        B = np.zeros((4, 2))
        B[2, 0] = self.DT
        B[3, 1] = self.DT * v / (self.WB * math.cos(steer) ** 2)

        # constant offset
        C = np.zeros((4, ))
        C[0] = self.DT * v * math.sin(theta) * theta
        C[1] = - self.DT * v * math.cos(theta) * theta
        C[3] = - self.DT * v * steer / (self.WB * math.cos(steer) ** 2)

        return A, B, C

    def load_raceline(self, path):
        """load the optimal raceline from the corresponding file"""

        tmp = np.loadtxt(path, delimiter=';', skiprows=3)
        return np.transpose(tmp[:, [1, 2, 5, 3]])

    def closest_raceline_point(self, x, v):
        """find the point on the raceline that is closest to the current state"""

        # determine search range
        dist_delta = np.sqrt(np.sum((self.raceline[[0, 1], 1] - self.raceline[[0, 1], 0]) ** 2, axis=0))
        ind_diff = np.ceil(v*self.DT/dist_delta) + 10

        ind_range = np.mod(np.arange(self.ind_prev, self.ind_prev + ind_diff), self.raceline.shape[1]).astype(int)

        # compute closest point
        ind = np.argmin(np.sum((self.raceline[0:2, ind_range]-x)**2, axis=0))
        self.ind_prev = ind_range[ind]

        return ind_range[ind]


""" --------------------------------------------------------------------------------------------------------------------
Main Program
-------------------------------------------------------------------------------------------------------------------- """

if __name__ == '__main__':

    # load the configuration for the desired Racetrack
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.8125}
    with open('config_Spielberg_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # create the simulation environment and initialize it
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # initialize the motion planner
    controller = MPC_Controller(conf.wpt_path)

    # initialize auxiliary variables
    laptime = 0.0
    control_lim = np.ceil(1 / (controller.freq * env.timestep)).astype(int)
    control_count = control_lim
    start = time.time()

    # main control loop
    while not done:

        # re-plan trajectory
        if control_count == control_lim:

            speed, steer = controller.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0],
                                           obs['linear_vels_x'][0])
            control_count = 0

        # update the simulation environment
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward

        env.render(mode='human_fast')

        # update counter
        control_count = control_count + 1

        if obs['lap_counts'] == 1:
            break

    # print results
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
