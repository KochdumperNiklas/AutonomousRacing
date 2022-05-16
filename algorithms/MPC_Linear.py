import numpy as np
import math
import cvxpy
from os.path import exists
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io import savemat
from auxiliary.process_lidar_data import process_lidar_data
from auxiliary.Polytope import Polytope
from auxiliary.Polygon import Polygon
from auxiliary.raceline import load_raceline
from auxiliary.raceline import get_reference_trajectory
from auxiliary.free_space import free_space

class MPC_Linear:
    """class representing an MPC controller that tracks the optimal raceline"""

    def __init__(self, params, settings):
        """object constructor"""

        # store algorithm settings
        settings['R'] = np.diag(settings['R'])
        settings['Rd'] = np.diag(settings['Rd'])
        settings['Q'] = np.diag(settings['Q'])
        settings['Qf'] = np.diag(settings['Qf'])

        settings['MAX_STEER'] = np.deg2rad(settings['MAX_STEER'])
        settings['MAX_DSTEER'] = np.deg2rad(settings['MAX_DSTEER'])

        self.settings = settings

        # load optimal raceline
        self.raceline = load_raceline(settings['path_raceline'])

        # wheelbase and length/widht of the car
        self.WB = params['lf'] + params['lr']
        self.width = params['width']
        self.length = params['length']

        # initialize previous control inputs and index of closest raceline point
        self.u_prev = np.zeros((2, self.settings['N']))
        self.ind_prev = 0

        # initialize counter for re-planning
        self.control_lim = np.ceil(1 / (settings['freq'] * 0.01)).astype(int)
        self.control_count = self.control_lim

    def plan(self, x, y, theta, v, scans):
        """plan a trajectory using MPC"""

        # check if control frequency is reached
        self.control_count += 1

        if self.control_count <= self.control_lim:
            return v + self.u_prev[0, 0] * self.settings['DT'], self.u_prev[1, 0]
        else:
            self.control_count = 0

        # get reference trajectory
        v = np.max((v, 0.1))
        ref_traj, ind = get_reference_trajectory(self.raceline, x, y, theta, v,
                                                 self.settings['N'], self.settings['DT'], self.ind_prev)
        self.ind_prev = ind

        # predict expected trajectory based on the previous control inputs (for linearization points)
        pred_traj = self.get_predicted_trajectory(x, y, theta, v)

        # compute drivable area
        drive_area = None

        if self.settings['DRIVE_AREA']:
            lidar_data = process_lidar_data(scans)
            drive_area = self.drivable_area(x, y, v, theta, lidar_data)

        # plan new trajectory using MPC
        x0 = np.array([x, y, v, theta])
        [u, traj] = self.mpc_optimization(x0, ref_traj, pred_traj, drive_area)

        if u is None:
            print('Failed to find a feasible solution!')
            u = self.u_prev

        self.u_prev = u

        # visualize the planned trajectory
        if self.settings['VISUALIZE']:
            self.visualization(x, y, theta, traj, scans, drive_area)

        return v + u[0, 0] * self.settings['DT'], u[1, 0]

    def mpc_optimization(self, x0, ref_traj, pred_traj, drive_area):
        """solve optimal control problem for MPC"""

        # initialization
        x = cvxpy.Variable((4, self.settings['N'] + 1))
        u = cvxpy.Variable((2, self.settings['N']))
        objective = 0.0
        constraints = []

        # objective function for optimal control problem
        for i in range(self.settings['N']):

            objective += cvxpy.quad_form(u[:, i], self.settings['R'])

            if i != 0:
                objective += cvxpy.quad_form(ref_traj[:, i] - x[:, i], self.settings['Q'])

            A, B, C = self.linearized_dynamic_function(pred_traj[2, i], pred_traj[3, i], 0)
            constraints += [x[:, i + 1] == A @ x[:, i] + B @ u[:, i] + C]

            if self.settings['DRIVE_AREA']:
                d = np.resize(drive_area[i].d, (drive_area[i].d.shape[0], ))
                s = cvxpy.Variable((len(d), ))
                constraints += [drive_area[i].c @ x[0:2, i + 1] <= d + s]
                constraints += [np.zeros((len(d), )) <= s]
                objective += cvxpy.quad_form(s, 200*np.identity(len(d)))

            if i < (self.settings['N'] - 1):
                objective += cvxpy.quad_form(u[:, i + 1] - u[:, i], self.settings['Rd'])
                constraints += [cvxpy.abs(u[1, i + 1] - u[1, i]) <= self.settings['MAX_DSTEER'] * self.settings['DT']]

        objective += cvxpy.quad_form(ref_traj[:, self.settings['N']] - x[:, self.settings['N']], self.settings['Qf'])

        # constraints for the optimal control problem
        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.settings['MAX_SPEED']]
        constraints += [x[2, :] >= self.settings['MIN_SPEED']]
        constraints += [cvxpy.abs(u[0, :]) <= self.settings['MAX_ACCEL']]
        constraints += [cvxpy.abs(u[1, :]) <= self.settings['MAX_STEER']]

        # solve the optimal control problem
        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            u = u.value
            traj = x.value
        else:
            u = None
            traj = None

        return u, traj

    def get_predicted_trajectory(self, x, y, theta, v):
        """predict trajectory based on the previous control inputs"""

        # initialization
        state = np.array([x, y, v, theta])
        pred_traj = np.zeros((4, self.settings['N']+1))
        pred_traj[:, 0] = state

        # loop over all time steps
        for i in range(self.u_prev.shape[1]):
            pred_traj[:, i+1] = pred_traj[:, i] + self.dynamic_function(pred_traj[:, i],
                                                                        self.u_prev[:, i]) * self.settings['DT']
            pred_traj[2, i+1] = np.max((np.min((pred_traj[2, i+1],
                                                self.settings['MAX_SPEED'])), self.settings['MIN_SPEED']))

        return pred_traj

    def drivable_area(self, x, y, v, theta, lidar_data):
        """compute the drivable area"""

        # convert lidar data to polygon of safe states
        pgon_safe = free_space(lidar_data)

        # initialize drivable area
        x_min = -self.length/2
        x_max = self.length/2
        y_min = -self.width
        y_max = self.width
        v_min = v
        v_max = v

        car = Polygon(np.array([x_min, x_min, x_max, x_max]),
                      np.array([-self.width, self.width, self.width, -self.width]))
        drive_area = []

        # loop over all time steps
        for i in range(1, self.settings['N']+1):

            # propagate drivable area forward in time
            v_max = min(self.settings['MAX_SPEED'], v_max + self.settings['MAX_ACCEL']*self.settings['DT'])
            v_min = max(self.settings['MIN_SPEED'], v_min - self.settings['MAX_ACCEL']*self.settings['DT'])
            x_max = x_max + v_max*self.settings['DT']
            x_min = x_min + v_min*self.settings['DT']
            y_max = y_max + v_max*np.sin(i*self.settings['MAX_STEER']*self.settings['DT'])*self.settings['DT']
            y_min = y_min - v_max*np.sin(i*self.settings['MAX_STEER']*self.settings['DT'])*self.settings['DT']

            pgon = Polygon(np.array([x_min, x_min, x_max, x_max]), np.array([y_min, y_max, y_max, y_min]))

            # compute intersection with set of safe states
            pgon_tmp = pgon & pgon_safe

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

        return drive_area

    def dynamic_function(self, x, u):
        """differential equation for the kinematic single track model"""

        u[1] = np.max((np.min((u[1], self.settings['MAX_STEER'])), -self.settings['MAX_STEER']))

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
        A[0, 2] = self.settings['DT'] * math.cos(theta)
        A[0, 3] = - self.settings['DT'] * v * math.sin(theta)
        A[1, 2] = self.settings['DT'] * math.sin(theta)
        A[1, 3] = self.settings['DT'] * v * math.cos(theta)
        A[3, 2] = self.settings['DT'] * math.tan(steer) / self.WB

        # input matrix B
        B = np.zeros((4, 2))
        B[2, 0] = self.settings['DT']
        B[3, 1] = self.settings['DT'] * v / (self.WB * math.cos(steer) ** 2)

        # constant offset
        C = np.zeros((4, ))
        C[0] = self.settings['DT'] * v * math.sin(theta) * theta
        C[1] = - self.settings['DT'] * v * math.cos(theta) * theta
        C[3] = - self.settings['DT'] * v * steer / (self.WB * math.cos(steer) ** 2)

        return A, B, C

    def visualization(self, x, y, theta, traj, scans, drive_area):
        """visualize the planned trajectory"""

        if not traj is None:
            lidar_data = process_lidar_data(scans)
            x_min = np.min(lidar_data[0, :])-2
            x_max = np.max(lidar_data[0, :])+2
            y_min = np.min(lidar_data[1, :])-2
            y_max = np.max(lidar_data[1, :])+2
            plt.cla()
            plt.plot(lidar_data[0, :], lidar_data[1, :], '.r', label='lidar measurements')
            rl = self.raceline[0:2, :] - np.array([[x], [y]])
            rl = np.dot(np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]), rl)
            plt.plot(rl[0, :], rl[1, :], 'g', label='optimal raceline')
            tr = traj[0:2, :] - np.array([[x], [y]])
            tr = np.dot(np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]), tr)
            if drive_area:
                for d in drive_area:
                    d.shift(-np.array([[x], [y]]))
                    d.rotate(-theta)
                    try:
                        d.plot('y')
                    except:
                        test = 1
                plt.plot([x_max + 1, x_max + 1], [y_max + 1, y_max + 1], 'y', label='driveable area')
            plt.plot(tr[0, :], tr[1, :], 'b', label='planned trajectory')
            plt.axis('equal')
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.legend(loc='upper right')
            plt.pause(0.1)