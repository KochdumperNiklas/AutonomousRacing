import numpy as np
import math
import cvxpy
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io import savemat
import largestinteriorrectangle as lir
from auxiliary.process_lidar_data import process_lidar_data
from auxiliary.Polytope import Polytope
from auxiliary.Polygon import Polygon
from auxiliary.raceline import load_raceline
from auxiliary.raceline import get_reference_trajectory
from auxiliary.free_space import free_space

class MPC_Linear:
    """class representing an MPC controller that tracks the optimal raceline"""

    def __init__(self, racetrack, params, path):
        """object constructor"""

        # load optimal raceline and controller settings
        self.raceline = load_raceline(path)
        self.load_controller_settings(racetrack)

        # wheelbase and width/length of the car
        self.WB = params['lf'] + params['lr']
        self.width = params['width']
        self.length = params['length']

        # initialize previous control inputs and index of closest raceline point
        self.u_prev = np.zeros((2, self.N))
        self.ind_prev = 0

    def plan(self, x, y, theta, v, scans):
        """plan a trajectory using MPC"""

        v = np.max((v, 0.1))

        print(v)

        # get reference trajectory
        ref_traj, ind = get_reference_trajectory(self.raceline, x, y, theta, v, self.N, self.DT, self.ind_prev)
        self.ind_prev = ind

        # predict expected trajectory based on the previous control inputs (for linearization points)
        pred_traj = self.get_predicted_trajectory(x, y, theta, v)

        # compute drivable area
        lidar_data = process_lidar_data(scans)
        drive_area = self.drivable_area2(x, y, v, theta, lidar_data)

        # plan new trajectory using MPC
        x0 = np.array([x, y, v, theta])
        self.u_prev = self.mpc_optimization(x0, ref_traj, pred_traj, drive_area)

        if self.u_prev is None:

            colors = ['r', 'b', 'g', 'r', 'b', 'g', 'r', 'b', 'g', 'r']

            for i in range(len(drive_area)):
                drive_area[i].plot(colors[i])

            p = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
                       lidar_data) + np.array([[x], [y]])
            plt.plot(p[0, :], p[1, :], '.r')
            plt.plot(x, y, '.k')
            plt.show()

        return v + self.u_prev[0, 0] * self.DT, self.u_prev[1, 0]

    def mpc_optimization(self, x0, ref_traj, pred_traj, drive_area):
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

            d = np.resize(drive_area[i].d, (drive_area[i].d.shape[0], ))
            s = cvxpy.Variable((len(d), ))
            constraints += [drive_area[i].c @ x[0:2, i + 1] <= d + s]
            constraints += [np.zeros((len(d), )) <= s]
            objective += cvxpy.quad_form(s, 200*np.identity(len(d)))

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
        prob.solve(solver=cvxpy.GUROBI, verbose=False, warm_start=True)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            u = u.value
        else:
            print("Error: Cannot solve mpc..")
            u = None

        return u

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

    def drivable_area2(self, x, y, v, theta, lidar_data):
        """compute the drivable area"""

        # convert lidar data to polygon of safe states
        lidar_data = lidar_data + np.array([[self.WB], [0]])
        pgon_safe = free_space(lidar_data)

        # initialize drivable area
        x_min = -self.length/2 + self.WB/2
        x_max = self.length/2 + self.WB/2
        y_min = -self.width
        y_max = self.width
        v_min = v
        v_max = v

        car = Polygon(np.array([x_min, x_min, x_max, x_max]),
                      np.array([-self.width/1, self.width/1, self.width/1, -self.width/1]))
        drive_area = []

        # loop over all time steps
        for i in range(1, self.N+1):

            # propagate drivable area forward in time
            x_max = x_max + v_max*self.DT + 0.5*self.MAX_ACCEL*(self.DT**2)
            x_min = x_min + v_min*self.DT - 0.5*self.MAX_ACCEL*(self.DT**2)
            v_max = v_max + self.MAX_ACCEL*self.DT
            v_min = v_min - self.MAX_ACCEL*self.DT
            y_max = y_max + v_max*np.sin(i*self.MAX_STEER*self.DT)*self.DT
            y_min = y_min - v_max*np.sin(i*self.MAX_STEER*self.DT)*self.DT

            pgon = Polygon(np.array([x_min, x_min, x_max, x_max]), np.array([y_min, y_max, y_max, y_min]))

            # compute intersection with set of safe states
            try:
                pgon_tmp = pgon & pgon_safe
            except:
                for p in drive_area:
                    try:
                        p.plot('r')
                    except:
                        est = 1
                pgon_safe.plot('b')
                test = 1

            # determine largest convex subset of the set
            pgon_drive = pgon_tmp.largest_convex_subset()

            # convert set to polytope
            poly = pgon_drive.polytope()

            # downsize the set by the vehicle dimensions
            poly = poly - car

            # transform from local to global coordinate frame
            poly.rotate(theta)
            poly.shift(np.array([[x], [y]]))

            drive_area.append(deepcopy(poly))

            # compute interval enclosure (= initial set for next iteration)
            x_min, x_max, y_min, y_max = pgon_drive.interval()

        """for p in drive_area:
            try:
                v = p.vertices()
            except:
                for p in drive_area:
                    try:
                        p.plot('b')
                    except:
                        test = 1

                p = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
                           lidar_data) + np.array([[x], [y]])
                plt.plot(p[0, :], p[1, :], '.r')
                plt.plot(x, y, '.k')
                plt.show()
                test = 1"""

        """for p in drive_area:
            p.plot('b')

        p = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), lidar_data) + np.array([[x], [y]])
        plt.plot(p[0, :], p[1, :], '.r')
        plt.plot(x, y, '.k')
        plt.show()"""

        return drive_area


    def drivable_area(self, x, y, v, theta, lidar_data):
        """compute the drivable area"""

        N = 20
        drive_area = []

        # initialize drivable area
        x_min = 0
        x_max = 0
        y_min = -self.width
        y_max = self.width
        v_min = v
        v_max = v

        # loop over all time steps
        for i in range(1, self.N+1):

            # propagate drivable area forward in time
            x_max = x_max + v_max*self.DT + 0.5*self.MAX_ACCEL*(self.DT**2)
            x_min = x_min + v_min*self.DT - 0.5*self.MAX_ACCEL*(self.DT**2)
            y_max = y_max + v_max*np.sin(i*self.MAX_STEER*self.DT)*self.DT
            y_min = y_min - v_max*np.sin(i*self.MAX_STEER*self.DT)*self.DT
            v_max = v_max + self.MAX_ACCEL*self.DT
            v_min = v_min - self.MAX_ACCEL*self.DT

            C = np.concatenate((np.eye(2), -np.eye(2)), axis=0)
            d = np.array([[x_max], [y_max], [-x_min], [-y_min]])
            poly = Polytope(C, d)

            # determine points that are inside the drivable area
            tmp = np.max(np.dot(poly.c, lidar_data) - np.dot(poly.d, np.ones((1, lidar_data.shape[1]))), axis=0)
            ind = [i for i in range(len(tmp)) if tmp[i] < 0]

            # intersect with the obstacles
            grid = np.ones((N, N))

            dx = (x_max - x_min)/N
            dy = (y_max - y_min)/N
            c_x = (x_max + x_min)/2
            c_y = (y_max + y_min)/2

            for j in ind:
                ind1 = np.floor((lidar_data[0, j] - c_x)/dx + N/2).astype(int)
                ind2 = np.floor((lidar_data[1, j] - c_y)/dy + N/2).astype(int)
                grid[min(ind2, N-1), min(ind1, N-1)] = 0

            rect = lir.lir(grid.astype(bool))

            x_min = c_x + (rect[0] - N/2)*dx
            x_max = c_x + (rect[0] + rect[2] - N/2)*dx
            y_min = c_y + (rect[1] - N/2)*dy
            y_max = c_y + (rect[1] + rect[3] - N/2)*dy

            poly_ = Polytope(poly.c, np.array([[x_max], [y_max-1.2*self.width/2], [-x_min], [-y_min-1.2*self.width/2]]))

            """# Debug
            poly.plot('r')
            poly_.plot('k')
            plt.plot(lidar_data[0, :], lidar_data[1, :], '.b')
            plt.plot(lidar_data[0, ind], lidar_data[1, ind], '.g')
            plt.show()"""

            # transform from local to global coordinate frame
            area = poly_
            area.rotate(theta)
            area.shift(np.array([[x], [y]]))

            drive_area.append(area)

        """colors = ['r', 'b', 'g', 'k', 'y', 'r', 'b', 'g', 'k', 'y']

        for i in range(len(drive_area)):
            drive_area[i].plot(colors[i])

        plt.plot(x, y, '.k')
        plt.show()"""

        return drive_area


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

    def load_controller_settings(self, racetrack):
        """load settings for the MPC controller"""

        # load controller settings from file
        path = 'racetracks/' + racetrack + '/settings_' + racetrack + '_MPC_Linear.txt'

        with open(path) as f:
            lines = f.readlines()

        for l in lines:
            exec(l, globals(), globals())

        # weighting matrices for MPC
        self.R = R                                                      # input cost matrix[accel, steer]
        self.Rd = Rd                                                    # input difference cost matrix[accel, steer]
        self.Q = Q                                                      # state cost matrix [x, y, v, yaw]
        self.Qf = Qf                                                    # final state cost matrix [x, y, v, yaw]

        # motion planning parameter
        self.DT = DT                                                    # time step size [s]
        self.N = N                                                      # number of time steps for MPC prediction
        self.freq = freq                                                # planning frequency [Hz]

        # input and state constraints
        self.MAX_STEER = MAX_STEER                                      # maximum steering angle [rad]
        self.MAX_DSTEER = MAX_DSTEER                                    # maximum steering speed [rad/s]
        self.MAX_SPEED = MAX_SPEED                                      # maximum speed [m/s]
        self.MIN_SPEED = MIN_SPEED                                      # minimum backward speed [m/s]
        self.MAX_ACCEL = MAX_ACCEL                                      # maximum acceleration [m/s**2]
