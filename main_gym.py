import time
import yaml
import gym
import numpy as np
from argparse import Namespace
from algorithms.MPC_Linear import MPC_Linear
from algorithms.ManeuverAutomaton import ManeuverAutomaton

CONTROLLER = 'ManeuverAutomaton'
RACETRACK = 'Spielberg'

if __name__ == '__main__':

    # load the configuration for the desired Racetrack
    path = 'racetracks/' + RACETRACK + '/config_' + RACETRACK + '.yaml'
    with open(path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    # create the simulation environment and initialize it
    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # initialize the motion planner
    if CONTROLLER == 'MPC_Linear':
        controller = MPC_Linear(RACETRACK, env.params, conf.wpt_path)
    elif CONTROLLER == 'ManeuverAutomaton':
        controller = ManeuverAutomaton(RACETRACK, env.params)
    else:
        raise Exception('Specified controller not available!')

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
                                           obs['linear_vels_x'][0], obs['scans'][0])
            control_count = 0

        # update the simulation environment
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward

        env.render(mode='human')

        # update counter and check if lap is finished
        control_count = control_count + 1

        if obs['lap_counts'] == 1:
            break

    # print results
    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time() - start)