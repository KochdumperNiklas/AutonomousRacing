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

MEASUREMENT = 'racetrack_1'
CONTROLLER = 'DisparityExtender'
RACETRACK = 'StonyBrook'



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
    settings = parse_settings(CONTROLLER, RACETRACK, True)
    params = car_parameter()
    exec('controller = ' + CONTROLLER + '(params, settings)')

    # initialization
    start_time = max(time_lidar[0], time_odom[0])
    time_lidar = time_lidar - start_time
    time_odom = time_odom - start_time
    time_control = time_control - start_time
    speed = []
    steer = []
    time = []
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

        # visualized the planned trajectory
        speed_, steer_ = controller.plan(None, None, None, v, lidar_data)

        # store data
        speed.append(speed_)
        steer.append(steer_)
        time.append(t)

        # update time
        t += 0.01

    # compare actual and expected control commands
    speed = np.asarray(speed)
    steer = np.asarray(steer)
    time = np.asarray(time)

    plt.close()
    plt.plot(time, speed, label='speed (simulation)')
    plt.plot(time_control, speed_real, label='speed (real)')
    plt.legend()
    plt.show()

    plt.plot(time, steer, label='steer (simulation)')
    plt.plot(time_control, steer_real, label='steer (real)')
    plt.legend()
    plt.show()
