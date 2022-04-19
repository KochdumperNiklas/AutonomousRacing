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
#import trajectory_planning_helpers.path_matching_global as tph
#import trajectory_planning_helpers.side_of_line as sol


#--------------------------- Controller Paramter ---------------------------
# System config
NX = 4          # state vector: z = [x, y, v, yaw]
NU = 2          # input vector: u = [accel, steer]
T = 10           # finite time horizon length

# MPC parameters
R = np.diag([0.01, 100.0])              # input cost matrix, penalty for inputs - [accel, steer]
Rd = np.diag([0.01, 100.0])             # input difference cost matrix, penalty for change of inputs - [accel, steer]
Q = np.diag([13.5, 13.5, 5.5, 13.0])       # state cost matrix, for the the next (T) prediction time steps [x, y, v, yaw]
Qf = np.diag([13.5, 13.5, 5.5, 13.0])      # state final matrix, penalty  for the final state constraints: [x, y, v, yaw]

# MPC prediction paramter
N_IND_SEARCH = 5                        # Search index number
DT = 0.10                               # time step [s]
dl = 0.20                               # dist step [m]

# Vehicle parameters
LENGTH = 0.58                       # Length of the vehicle [m]
WIDTH = 0.31                        # Width of the vehicle [m]
WB = 0.33                           # Wheelbase [m]
MAX_STEER = np.deg2rad(24.0)        # maximum steering angle [rad]
#MAX_STEER = np.deg2rad(24.0)        # maximum steering angle [rad]         #REAL PARAMETER
MAX_DSTEER = np.deg2rad(180.0)       # maximum steering speed [rad/s]
#MAX_DSTEER = np.deg2rad(180.0)       # maximum steering speed [rad/s]      #REAL PARAMETER
MAX_SPEED = 6                   # maximum speed [m/s]
MIN_SPEED = 0                       # minimum backward speed [m/s]
MAX_ACCEL = 2.5                     # maximum acceleration [m/ss]


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1, :] + (t * diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment


@njit(fastmath=False, cache=True)
def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi
    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle

class State:
    """
    vehicle state class
    """
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class Controller:

    def __init__(self, conf, wb):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.mpc_initialize = 0
        self.target_ind = 0
        self.odelta = None
        self.oa = None
        self.origin_switch = 1

    def load_waypoints(self, conf):
        # Loading the x and y waypoints in the "..._raceline.vsv" that include the path to follow
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)


    def calc_nearest_index(self, state, cx, cy, cyaw, pind):
        """
        calc index of the nearest ref trajector in N steps
        :param node: path information X-Position, Y-Position, current index.
        :return: nearest index,
        """

        if pind == len(cx)-1:
            dx = [state.x - icx for icx in cx[0:(0 + N_IND_SEARCH)]]
            dy = [state.y - icy for icy in cy[0:(0 + N_IND_SEARCH)]]

            d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
            mind = min(d)
            ind = d.index(mind) + 0

        else:
            dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
            dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

            d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
            mind = min(d)
            ind = d.index(mind) + pind

        mind = math.sqrt(mind)
        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y
        angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind


    def calc_ref_trajectory(self, state, cx, cy, cyaw, sp, dl, pind):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        # Create placeholder Arrays for the reference trajectory for T steps
        ref_traj = np.zeros((NX, T + 1))
        dref = np.zeros((1, T + 1))
        ncourse = len(cx)

        # Find nearest index/setpoint from where the trajectories are calculated
        ind, _ = Controller.calc_nearest_index(self, state, cx, cy, cyaw, pind)

        #if pind >= ind:
        #    ind = pind

        # Load the initial parameters from the setpoint into the trajectory
        ref_traj[0, 0] = cx[ind]
        ref_traj[1, 0] = cy[ind]
        ref_traj[2, 0] = sp[ind]
        ref_traj[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0                # steer operational point should be 0

        # Initialize Parameter
        travel = 0.0
        self.origin_switch = 1

        for i in range(T + 1):
            travel += abs(state.v) * DT     # Travel Distance into the future based on current velocity: s= v * t
            dind = int(round(travel / dl))  # Number of distance steps we need to look into the future

            if (ind + dind) < ncourse:
                ref_traj[0, i] = cx[ind + dind]
                ref_traj[1, i] = cy[ind + dind]
                ref_traj[2, i] = sp[ind + dind]

                # IMPORTANT: Take Care of Heading Change from 2pi -> 0 and 0 -> 2pi, so that all headings are the same
                if cyaw[ind + dind] -state.yaw > 5:
                    ref_traj[3, i] = abs(cyaw[ind + dind] -2* math.pi)
                elif cyaw[ind + dind] -state.yaw < -5:
                    ref_traj[3, i] = abs(cyaw[ind + dind] + 2 * math.pi)
                else:
                    ref_traj[3, i] = cyaw[ind + dind]

            else:
                # This function takes care about the switch at the origin/ Lap switch
                ref_traj[0, i] = cx[self.origin_switch]
                ref_traj[1, i] = cy[self.origin_switch]
                ref_traj[2, i] = sp[self.origin_switch]
                ref_traj[3, i] = cyaw[self.origin_switch]
                dref[0, i] = 0.0
                self.origin_switch = self.origin_switch +1

        return ref_traj, ind, dref

    def predict_motion(x0, oa, od, xref):
        path_predict = xref * 0.0
        for i, _ in enumerate(x0):
            path_predict[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, T + 1)):
            state = Controller.update_state(state, ai, di)
            path_predict[0, i] = state.x
            path_predict[1, i] = state.y
            path_predict[2, i] = state.v
            path_predict[3, i] = state.yaw

        return path_predict

    def update_state(state, a, delta):

        # input check
        if delta >= MAX_STEER:
            delta = MAX_STEER
        elif delta <= -MAX_STEER:
            delta = -MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * DT
        state.y = state.y + state.v * math.sin(state.yaw) * DT
        state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
        state.v = state.v + a * DT

        if state.v > MAX_SPEED:
            state.v = MAX_SPEED
        elif state.v < MIN_SPEED:
            state.v = MIN_SPEED

        return state

    @njit(fastmath=False, cache=True)
    def get_kinematic_model_matrix(v, phi, delta):
        """
        ************ Single Track Model: Linear - Kinematic ********
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax +Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """

        # State (or system) matrix A, 4x4
        A = np.zeros((NX, NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = DT * math.cos(phi)
        A[0, 3] = - DT * v * math.sin(phi)
        A[1, 2] = DT * math.sin(phi)
        A[1, 3] = DT * v * math.cos(phi)
        A[3, 2] = DT * math.tan(delta) / WB

        # Input Matrix B; 4x2
        B = np.zeros((NX, NU))
        B[2, 0] = DT
        B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

        C = np.zeros(NX)
        C[0] = DT * v * math.sin(phi) * phi
        C[1] = - DT * v * math.cos(phi) * phi
        C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

        return A, B, C

    def get_nparray_from_matrix(x):
        return np.array(x).flatten()

    def linear_mpc_control(ref_path, x0, dref, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param a_old: acceleration of T steps of last time
        :param delta_old: delta of T steps of last time
        :return: acceleration and delta strategy based on current information
        """

        if oa is None or od is None:
            oa = [0.0] * T
            od = [0.0] * T

        # Call the Motion Prediction function: Predict the vehicle motion for x-steps
        path_predict = Controller.predict_motion(x0, oa, od, ref_path)
        poa, pod = oa[:], od[:]

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v = Controller.mpc_optimization(ref_path, path_predict, x0, dref)

        # Calculta the u change value
        du = sum(abs(mpc_a - poa)) + sum(abs(mpc_delta - pod))

        return mpc_a, mpc_delta, mpc_x, mpc_y, mpc_yaw, mpc_v, path_predict

    def mpc_optimization(ref_traj, path_predict, x0, dref):
        """
        Create and solve the quadratic optimization problem using cvxpy, solver: OSQP
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html

        xref: reference trajectory (desired trajectory: [x, y, v, yaw])
        path_predict: predicted states in T steps
        x0: initial state
        dref: reference steer angle
        :return: optimal acceleration and steering strateg
        """

        # Initialize vectors
        x = cvxpy.Variable((NX, T + 1))     # Vehicle State Vector
        u = cvxpy.Variable((NU, T))         # Control Input vector
        objective = 0.0                     # Objective value of the optimization problem, set to zero
        constraints = []                    # Create constraints array

        # Formulate and create the finite-horizon optimal control problem (objective function)
        for t in range(T):
            objective += cvxpy.quad_form(u[:, t], R)

            if t != 0:
                objective += cvxpy.quad_form(ref_traj[:, t] - x[:, t], Q)

            A, B, C = Controller.get_kinematic_model_matrix(path_predict[2, t], path_predict[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            if t < (T - 1):
                objective += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

        objective+= cvxpy.quad_form(ref_traj[:, T] - x[:, T], Qf)

        # Create the constraints (upper and lower bounds of states and inputs) for the optimization problem
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= MAX_SPEED]
        constraints += [x[2, :] >= MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)


        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = Controller.get_nparray_from_matrix(x.value[0, :])
            oy = Controller.get_nparray_from_matrix(x.value[1, :])
            ov = Controller.get_nparray_from_matrix(x.value[2, :])
            oyaw = Controller.get_nparray_from_matrix(x.value[3, :])
            oa = Controller.get_nparray_from_matrix(u.value[0, :])
            odelta = Controller.get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov

    def MPC_Controller (self, vehicle_state, path):

        # --------------------------- Inititalize ---------------------------
        # Initialize the MPC parameter
        if self.mpc_initialize == 0:
            #self.target_ind, _ = Controller.calc_nearest_index(vehicle_state, cx, cy, cyaw, 0)
            self.target_ind = 0
            self.odelta, self.oa = None, None
            self.mpc_initialize = 1

        #------------------- MPC CONTROL LOOP ---------------------------------
        # Extract information about the trajectory that needs to be followed
        cx = path[0]        # Trajectory x-Position
        cy = path[1]        # Trajectory y-Position
        cyaw = path[2]      # Trajectory Heading angle
        sp = path[4]        # Trajectory Velocity

        # Calculate the next reference trajectory for the next T steps:: [x, y, v, yaw]
        ref_path, self.target_ind, ref_delta = Controller.calc_ref_trajectory(self, vehicle_state, cx, cy, cyaw, sp, dl, self.target_ind)

        # Create state vector based on current vehicle state: x-position, y-position,  velocity, heading
        x0 = [vehicle_state.x, vehicle_state.y, vehicle_state.v, vehicle_state.yaw]

        # Solve the Linear MPC Control problem
        self.oa, self.odelta, ox, oy, oyaw, ov, path_predict = Controller.linear_mpc_control(ref_path, x0, ref_delta, self.oa, self.odelta)

        if self.odelta is not None:
            di, ai = self.odelta[0], self.oa[0]

        ###########################################
        #                    DEBUG
        ##########################################

        debugplot = 0
        if debugplot == 1:
            plt.cla()
            # plt.axis([-40, 2, -10, 10])
            plt.axis([vehicle_state.x - 6, vehicle_state.x + 4.5, vehicle_state.y - 2.5, vehicle_state.y  + 2.5])
            plt.plot(self.waypoints[:, [1]], self.waypoints[:, [2]], linestyle='solid', linewidth=2, color='#005293', label='Raceline')
            plt.plot(vehicle_state.x, vehicle_state.y, marker='o', markersize=10, color='red', label='CoG')
            plt.plot(ref_path[0], ref_path[1], linestyle='dotted', linewidth=8, color='purple',label='MPC Input: Ref. Trajectory for T steps')
            #plt.plot(cx[self.target_ind], cy[self.target_ind], marker='x', markersize=10, color='green',)
            plt.plot(ox, oy, linestyle='dotted', linewidth=5, color='green',label='MPC Output: Trajectory for T steps')
            plt.legend()
            plt.pause(0.001)
            plt.axis('equal')

        debugplot2 = 0
        if debugplot2 == 1:
            plt.cla()
            # Creating the number of subplots
            fig, axs = plt.subplots(3, 1)
            #  Velocity of the vehicle
            axs[0].plot(ov, linestyle='solid', linewidth=2, color='#005293')
            axs[0].set_ylim([0, max(ov) + 0.5])
            axs[0].set(ylabel='Velocity in m/s')
            axs[0].grid(axis="both")

            axs[1].plot(self.oa, linestyle='solid', linewidth=2, color='#005293')
            axs[1].set_ylim([0, max(self.oa) + 0.5])
            axs[1].set(ylabel='Acceleration in m/s')
            axs[1].grid(axis="both")
            plt.pause(0.001)
            plt.axis('equal')

        ###########################################
        #                    DEBUG
        ##########################################


        #------------------- MPC CONTROL Output ---------------------------------
        # Create the final steer and speed parameter that need to be sent out

        # Steering Output: First entry of the MPC steering angle output vector in degree
        steer_output = self.odelta[0]

        # Acceleration Output: First entry of the MPC acceleration output in m/s2
        # The F1TENTH Gym needs velocity as an control input: Acceleration -> Velocity
        # accelerate
        #speed_output=self.oa[0]*T*DT
        #speed_output=ref_path[2][1]*0.50
        speed_output= vehicle_state.v + self.oa[0] * DT

        #print ("Current Speed:", vehicle_state.v, "Control Speed:", speed,"MPC Speed:",ov)
        #print("Vehicle Heading: ", vehicle_state.yaw, "MPC Heading:",oyaw[0], "RefPath Heading:",ref_path[3], "Racline Heading:",cyaw[self.target_ind])
        #print ("Vehicle X:", vehicle_state.x, "Target X:",cx[self.target_ind],"----- Vehicle Y:", vehicle_state.y, " Target Y:",cy[self.target_ind])

        u = np.concatenate((np.resize(self.oa, (1, self.oa.shape[0])), np.resize(self.odelta, (1, self.odelta.shape[0]))), axis=0)

        return speed_output, steer_output, u, ref_path, path_predict



class LatticePlanner:

    def __init__(self, conf, env, wb):
        self.conf = conf                        # Current configuration for the gym based on the maps
        self.env = env                          # Current environment parameter
        self.load_waypoints(conf)               # Waypoints of the raceline
        self.init_flag = 0                      # Initialization of the states
        self.calcspline = 0                     # Flag for Calculation the Cubic Spline
        self.initial_state = []

    def load_waypoints(self, conf):
        """
        Loading the x and y waypoints in the "..._raceline.csv" which includes the path to follow
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def plan(self):
        """
        Loading the individual data from the global, optimal raceline and creating one list
        """

        cx = self.waypoints[:, 1]       # X-Position of Raceline
        cy = self.waypoints[:, 2]       # Y-Position of Raceline
        cyaw = self.waypoints[:, 3]     # Heading on Raceline
        ck = self.waypoints[:, 4]       # Curvature of Raceline
        cv = self.waypoints[:, 5]       # velocity on Raceline

        global_raceline = [cx, cy, cyaw, ck, cv]

        return global_raceline

    def control(self, pose_x, pose_y, pose_theta, velocity, path):
        """
        Control loop for calling the controller
        """

        # -------------------- INITIALIZE Controller ----------------------------------------
        if self.init_flag == 0:
            vehicle_state = State(x=pose_x, y=pose_y, yaw=pose_theta, v=0.1)
            self.init_flag = 1
        else:
            vehicle_state = State(x=pose_x, y=pose_y, yaw=pose_theta, v=velocity)

        # -------------------- Call the MPC Controller ----------------------------------------
        speed, steering_angle, u, ref_path, path_pred = controller.MPC_Controller(vehicle_state, path)

        return speed, steering_angle, u, ref_path, path_pred

""" --------------------------------------------------------------------------------------------------------------------
Auxiliary Functions
-------------------------------------------------------------------------------------------------------------------- """

def load_raceline(path):
    """load the optimal raceline from the corresponding file"""

    tmp = np.loadtxt(path, delimiter=';', skiprows=3)
    return np.transpose(tmp[:, [1, 2, 5, 3]])

def closest_point(points, p):
    """finds the closest point to p in the given point cloud"""

    return np.argmin(np.sum((points-p)**2, axis=0))




""" --------------------------------------------------------------------------------------------------------------------
Controller
-------------------------------------------------------------------------------------------------------------------- """

class MPC_Controller:
    """class representing an MPC controller that tracks the optimal raceline"""

    def __init__(self, path):
        """object constructor"""

        self.raceline = load_raceline(path)                             # load optimal raceline from file
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

        return v + self.u_prev[0, 0] * self.DT, self.u_prev[1, 0], self.u_prev, ref_traj, pred_traj


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
            u1 = Controller.get_nparray_from_matrix(u.value[0, :])
            u2 = Controller.get_nparray_from_matrix(u.value[1, :])
            u = np.concatenate((np.resize(u1, (1, self.N)), np.resize(u2, (1, self.N))), axis=0)
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

    # Load the configuration for the desired Racetrack
    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 0.8125}
    with open('config_Spielberg_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # Dictionary for changing vehicle paramters for the vehicle dynamics in the F1TENTH Gym
    params_dict = {'mu': 1.0489,
                   'C_Sf': 4.718,
                   'C_Sr': 5.4562,
                   'lf': 0.15875,
                   'lr': 0.17145,
                   'h': 0.074,
                   'm': 3.74,
                   'I': 0.04712,
                   's_min': -0.4189,
                   's_max': 0.4189,
                   'sv_min': -3.2,
                   'sv_max': 3.2,
                   'v_switch': 7.319,
                   'a_max': 9.51,
                   'v_min': -5.0,
                   'v_max': 20.0,
                   'width': 0.31,
                   'length': 0.58}

    # Create the simulation environment and inititalize it
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1,params=params_dict)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy+0.5, conf.stheta]]))
    env.render()

    # Creating the Motion planner and Controller object that is used in Gym
    planner = LatticePlanner(conf, env, 0.17145 + 0.15875)
    controller = Controller(conf, 0.17145 + 0.15875)
    test = MPC_Controller(conf.wpt_path)
    #speed, steer = test.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0])

    # Initialize Simulation
    laptime = 0.0
    control_count = 10
    start = time.time()

    # Load global raceline to create a path variable that includes all reference path information
    path = planner.plan()

    # -------------------------- SIMULATION LOOP  ----------------------------------------
    while not done:
        # Call the function for planning a path, only every 15th timestep
        if control_count == 10:

            # Call the function for tracking speed and steering
            # MPC specific: We solve the MPC problem only every 6th timestep of the simulation to decrease the sim time
            start = time.time()
            speed_, steer_, u_, ref_, pred_ = planner.control(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0], path)
            print(time.time() - start)
            start = time.time()
            speed, steer, u, ref, pred = test.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0])
            print(time.time() - start)
            control_count = 0

            if abs(speed-speed_) > 0.001 or abs(steer - steer_) > 0.001:
                temp = 1

        # Update the simulation environment
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward

        env.render(mode='human_fast')

        # Update Asynchronous Counter for the MPC loop
        control_count = control_count + 1

        if obs['lap_counts'] == 1:
            break
    if conf_dict['logging'] == 'True':
        pickle.dump(logging, open("../Data_Visualization/datalogging_MPC_KS.p", "wb"))

    # Print Statement that simulation is over
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)
