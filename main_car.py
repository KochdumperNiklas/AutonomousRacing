import rospy
import numpy as np
from auxiliary.parse_settings import parse_settings
from algorithms import *
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

CONTROLLER = 'ManeuverAutomaton'
RACETRACK = 'Oschersleben'

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
    param['s_min'] = -0.4189        # minimum steering angle[rad]
    param['s_max'] = 0.4189         # maximum steering angle[rad]
    param['sv_min'] = -3.2          # minimum steering velocity[rad / s]
    param['sv_max'] = 3.2           # maximum steering velocity[rad / s]

    # longitudinal constraints
    param['v_min'] = -5.0           # minimum velocity[m / s]
    param['v_max'] = 20.0           # maximum velocity[m / s]
    param['v_switch'] = 7.319       # switching velocity[m / s]
    param['a_max'] = 9.51           # maximum absolute acceleration[m / s ^ 2]

    # size of the car
    param['width'] = 0.31
    param['length'] = 0.58

    return param

class PublisherSubscriber:
    """wrapper class that handles writing control commands and reading sensor measurements"""

    def __init__(self, planner):
        """class constructor"""

        # publisher
        self.pub = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/teleop", AckermannDriveStamped,
                                   queue_size=1)

        # subscribers
        self.sub_lidar = rospy.Subscriber("/scan", LaserScan, self.callback_lidar)
        self.sub_velocity = rospy.Subscriber("/vesc/odom/", Odometry, self.callback_velocity)

        # store motion planner
        self.controller = controller

        # wait until first measurement is obtained
        rate = rospy.Rate(1000)

        while not hasattr(self, 'lidar_data') or not hasattr(self, 'velocity'):
            rate.sleep()

        # start timers for control command publishing and re-planning
        self.timer1 = rospy.Timer(rospy.Duration(0.001), self.callback_timer1)
        self.timer2 = rospy.Timer(rospy.Duration(0.01), self.callback_timer2)
        rospy.spin()

    def callback_lidar(self, msg):
        """store lidar data"""

        self.lidar_data = msg.ranges

    def callback_velocity(self, msg):
        """calculate absolute velocity from x- any y-components"""

        self.velocity = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)

    def callback_timer1(self, timer):
        """publish the current control commands"""

        msg = AckermannDriveStamped()
        msg.drive.speed = self.u[1, 0]
        msg.drive.steering_angle = self.u[0, 0]

        self.pub.publish(msg)

    def callback_timer2(self, timer):
        """obtain new control commands from the controller"""

        self.u = controller.plan(None, None, None, self.velocity, self.ranges)

if __name__ == '__main__':
    """main entry point"""

    # initialize the motion planner
    params = car_parameter()
    settings = parse_settings(CONTROLLER, RACETRACK, False)

    if CONTROLLER == 'MPC_Linear':
        controller = MPC_Linear(params, settings)
    elif CONTROLLER == 'ManeuverAutomaton':
        controller = ManeuverAutomaton(params, settings)
    else:
        raise Exception('Specified controller not available!')

    # start control cycle
    PublisherSubscriber(controller)
