import matplotlib.pyplot as plt
import pandas as pd
import os.path
import numpy as np
from auxiliary.parse_settings import parse_settings
from algorithms.ManeuverAutomaton import ManeuverAutomaton
from algorithms.MPC_Linear import MPC_Linear
from algorithms.GapFollower import GapFollower
from algorithms.DisparityExtender import DisparityExtender
from algorithms.SwitchingDriver import SwitchingDriver
from auxiliary.process_lidar_data import process_lidar_data
from auxiliary.vehicle_model import simulate
from localization.IterativeClosestLine import IterativeClosestLine
from localization.ParticleFilter import ParticleFilter

MEASUREMENT = 'racetrack_6'
CONTROLLER = 'GapFollower'
RACETRACK = 'StonyBrook'
OBSERVER = 'ParticleFilter'
VISUALIZE = False
#x0 = np.array([-6, -5, -0.15])
x0 = np.array([-0.63, -0.3, -3.11])

def car_parameter():
    """parameter for the car"""

    param = {}

    param['mu'] = 1.0489
    param['C_Sf'] = 4.718
    param['C_Sr'] = 5.4562
    param['lf'] = 0.15875
    param['lr'] = 0.17145
    param['h'] = 0.074
    param['m'] = 3.74
    param['I'] = 0.04712

    # steering constraints
    param['s_min'] = -0.4189  # minimum steering angle[rad]
    param['s_max'] = 0.4189  # maximum steering angle[rad]
    param['sv_min'] = -3.2  # minimum steering velocity[rad / s]
    param['sv_max'] = 3.2  # maximum steering velocity[rad / s]

    # longitudinal constraints
    param['v_min'] = -5.0  # minimum velocity[m / s]
    param['v_max'] = 20.0  # maximum velocity[m / s]
    param['v_switch'] = 7.319  # switching velocity[m / s]
    param['a_max'] = 9.51  # maximum absolute acceleration[m / s ^ 2]

    # size of the car
    param['width'] = 0.31
    param['length'] = 0.58
    param['lidar'] = 0.1

    return param


if __name__ == '__main__':

    # read measurement files (lidar)
    dirpath = os.path.dirname(os.path.abspath(__file__))
    lidarpath = os.path.join(dirpath, 'measurements', MEASUREMENT, 'lidar.csv')
    lidar = pd.read_csv(lidarpath)
    time_lidar = np.asarray(lidar['%time']) * 10**(-9)
    lidar = np.asarray(lidar)
    lidar = lidar[:, 11:1091]

    # read measurement files (velocity)
    odompath = os.path.join(dirpath, 'measurements', MEASUREMENT, 'odometry.csv')
    odom = pd.read_csv(odompath)
    vel = np.asarray(odom['field.twist.twist.linear.x'])
    time_odom = np.asarray(odom['%time']) * 10**(-9)

    # read measurement files (control commands)
    controlpath = os.path.join(dirpath, 'measurements', MEASUREMENT, 'control.csv')
    control = pd.read_csv(controlpath)
    speed_real = np.asarray(control['field.drive.speed'])
    steer_real = np.asarray(control['field.drive.steering_angle'])
    time_control = np.asarray(control['%time']) * 10**(-9)

    # initialize motion planner
    settings = parse_settings(CONTROLLER, RACETRACK, VISUALIZE)
    params = car_parameter()
    exec('controller = ' + CONTROLLER + '(params, settings)')

    # initialize localization algorithm
    settings = parse_settings(OBSERVER, RACETRACK, VISUALIZE)
    exec('observer = ' + OBSERVER + '(params, settings, x0[0], x0[1], x0[2])')

    # initialization
    start_time = max(time_lidar[0], time_odom[0])
    time_lidar = time_lidar - start_time
    time_odom = time_odom - start_time
    time_control = time_control - start_time
    speed = []
    steer = []
    time = []
    traj_ = []
    t = 0

    # main control loop
    while t < time_lidar[len(time_lidar)-1]:

        # update lidar data and velocity
        ind = np.where(time_lidar <= t)
        ind = ind[0]
        lidar_data = lidar[ind[len(ind)-1], :]

        ind = np.where(time_odom <= t)
        ind = ind[0]
        v = vel[ind[len(ind)-1]]

        # visualize the planned trajectory
        speed_, steer_ = controller.plan(None, None, None, v, lidar_data)

        x, y, theta = observer.localize(lidar_data, v, steer_, speed_)
        traj_.append(np.array([x, y, theta]))

        # store data
        speed.append(speed_)
        steer.append(steer_)
        time.append(t)

        # update time
        t += 0.01

    # simulate the trajectory that the car drove
    traj = [np.array([x0[0], x0[1], 0.0, 0.0, -3.5, 0.0, 0.0])]

    for i in range(len(time_control)-1):
        u = np.array([1*speed_real[i], 0.7*steer_real[i]])
        tmp = simulate(traj[-1], u, np.array([0, time_control[i+1] - time_control[i]]), params)
        traj.append(tmp[1])

    # compare actual and expected control commands
    """speed = np.asarray(speed)
    steer = np.asarray(steer)
    time = np.asarray(time)
    traj = np.asarray(traj)
    traj_ = np.asarray(traj_)

    plt.close()
    plt.plot(time, speed, label='speed (simulation)')
    plt.plot(time_control, speed_real, label='speed (real)')
    plt.legend()
    plt.show()

    plt.plot(time, steer, label='steer (simulation)')
    plt.plot(time_control, steer_real, label='steer (real)')
    plt.legend()
    plt.show()"""

    traj_ = np.asarray(traj_)
    #plt.plot(traj[:, 0], traj[:, 1], 'r')
    plt.plot(traj_[:, 0], traj_[:, 1], 'b')
    #for m in observer.map:
    #    m.plot('g')
    plt.show()
    test = 1