import rospy
import numpy as np
from sshkeyboard import listen_keyboard, stop_listening
from auxiliary.parse_settings import parse_settings
from algorithms.ManeuverAutomaton import ManeuverAutomaton
from algorithms.MPC_Linear import MPC_Linear
from algorithms.GapFollower import GapFollower
from algorithms.DisparityExtender import DisparityExtender
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

CONTROLLER = 'ManeuverAutomaton'
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
    param['s_min'] = -0.4189        # minimum steering angle [rad]
    param['s_max'] = 0.4189         # maximum steering angle [rad]
    param['sv_min'] = -3.2          # minimum steering velocity [rad / s]
    param['sv_max'] = 3.2           # maximum steering velocity [rad / s]

    # longitudinal constraints
    param['v_min'] = -5.0           # minimum velocity [m / s]
    param['v_max'] = 20.0           # maximum velocity [m / s]
    param['v_switch'] = 7.319       # switching velocity [m / s]
    param['a_max'] = 9.51           # maximum absolute acceleration [m / s ^ 2]

    # size of the car
    param['width'] = 0.31
    param['length'] = 0.58
    param['lidar'] = 0.1

    return param

class PublisherSubscriber:
    """wrapper class that handles writing control commands and reading sensor measurements"""

    def __init__(self, controller):
        """class constructor"""

        # publisher
        self.pub = rospy.Publisher("/vesc/low_level/ackermann_cmd_mux/input/teleop", AckermannDriveStamped,
                                   queue_size=1)

        # subscribers
        self.sub_lidar = rospy.Subscriber("/scan", LaserScan, self.callback_lidar)
        self.sub_velocity = rospy.Subscriber("/vesc/odom/", Odometry, self.callback_velocity)

        # store motion planner
        self.controller = controller

        # initialize control input and auxiliary variables
        self.u = np.array([0.0, 0.0])
        self.run = False
        self.init = True

        # wait until first measurement is obtained
        rate = rospy.Rate(1000)

        while not hasattr(self, 'lidar_data') or not hasattr(self, 'velocity'):
            rate.sleep()

        # start timers for control command publishing and re-planning
        self.timer1 = rospy.Timer(rospy.Duration(0.001), self.callback_timer1)
        self.timer2 = rospy.Timer(rospy.Duration(0.01), self.callback_timer2)
        self.timer3 = rospy.Timer(rospy.Duration(0.001), self.callback_timer3)
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

        if self.run:   
            msg.drive.speed = self.u[0]
            msg.drive.steering_angle = 1.4*self.u[1]
        else:
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0

        self.pub.publish(msg)

    def callback_timer2(self, timer):
        """obtain new control commands from the controller"""

        self.u = self.controller.plan(None, None, None, self.velocity, self.lidar_data)

    def callback_timer3(self, timer):
        """start keyboard listener in new thread"""

        if self.init:
           listen_keyboard(on_press=self.key_press)
           self.init = False

    def key_press(self, key):
        """detect keyboard commands"""

        if key == "s":
           self.run = True
        elif key == "e":
           self.run = False
        elif key == "k":
           stop_listening()


def start_controller():

    # initialize the motion planner
    params = car_parameter()
    settings = parse_settings(CONTROLLER, RACETRACK, False)
    exec('controller = ' + CONTROLLER + '(params, settings)')

    # start control cycle
    PublisherSubscriber(locals()['controller'])
