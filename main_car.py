import rospy
import time
import numpy as np
from sshkeyboard import listen_keyboard, stop_listening
from auxiliary.parse_settings import parse_settings
from algorithms.ManeuverAutomaton import ManeuverAutomaton
from algorithms.MPC_Linear import MPC_Linear
from algorithms.GapFollower import GapFollower
from algorithms.DisparityExtender import DisparityExtender
from algorithms.PurePersuit import PurePersuit
from localization.ParticleFilter import ParticleFilter
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Bool

CONTROLLER = 'ManeuverAutomaton'
RACETRACK = 'StonyBrook'
OBSERVER = 'ParticleFilter'
x0 = np.array([0, 0, -3.14])

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
        self.sub_observer = rospy.Subscriber("observer", Float32MultiArray, self.callback_observer)
        self.sub_keyboard = rospy.Subscriber("keyboard", Bool, self.callback_keyboard)

        # store motion planner and observer
        self.controller = controller

        # initialize control input and auxiliary variables
        self.u = np.array([0.0, 0.0])
        self.run = False
        self.x = x0[0]
        self.y = x0[1]
        self.theta = x0[2]

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

        self.lidar_data = np.asarray(msg.ranges)[0:1080]

    def callback_velocity(self, msg):
        """calculate absolute velocity from x- any y-components"""

        self.velocity = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)

    def callback_observer(self, msg):
        """store the current pose estimates"""

        self.x = msg.data[0]
        self.y = msg.data[1]
        self.theta = msg.data[2]

    def callback_keyboard(self, msg):
        """store the current keyboard commands"""

        self.run = msg.data

    def callback_timer1(self, timer):
        """publish the current control commands"""

        msg = AckermannDriveStamped()

        if self.run and np.mean(self.lidar_data[500:580]) > 0.2:   
            msg.drive.speed = self.u[0]
            msg.drive.steering_angle = self.u[1]
        else:
            msg.drive.speed = 0.0
            msg.drive.steering_angle = 0.0

        self.pub.publish(msg)

    def callback_timer2(self, timer):
        """obtain new control commands from the controller"""

        self.u = self.controller.plan(self.x, self.y, self.theta, self.velocity, self.lidar_data)


class Observer:

    def __init__(self, observer):
        """class constructor"""

        # publisher
        self.pub = rospy.Publisher("observer", Float32MultiArray, queue_size=1)

        # subscribers
        self.sub_lidar = rospy.Subscriber("/scan", LaserScan, self.callback_lidar)
        self.sub_velocity = rospy.Subscriber("/vesc/odom/", Odometry, self.callback_velocity)
        self.sub_control = rospy.Subscriber("/vesc/low_level/ackermann_cmd_mux/input/teleop", AckermannDriveStamped, self.callback_commands)
      
        # store observer
        self.observer = observer

        # initialize control input and auxiliary variables
        self.u = np.array([0.0, 0.0])
        self.x = observer.state[0][0]
        self.y = observer.state[0][1]
        self.theta = observer.state[0][4]

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

        self.lidar_data = np.asarray(msg.ranges)[0:1080]

    def callback_velocity(self, msg):
        """calculate absolute velocity from x- any y-components"""

        self.velocity = np.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)

    def callback_commands(self, msg):
        """store current control commands"""

        self.u = np.array([msg.drive.speed, msg.drive.steering_angle])

    def callback_timer1(self, timer):
        """publish the current pose estimate"""

        msg = Float32MultiArray()
        msg.data = [self.x, self.y, self.theta]

        self.pub.publish(msg)

    def callback_timer2(self, timer):
        """update pose estimate"""

        self.x, self.y, self.theta = self.observer.localize(self.lidar_data, self.velocity, self.u[1], self.u[0])


class Keyboard:

    def __init__(self):
        """class constructor"""

        # publisher
        self.pub = rospy.Publisher("keyboard", Bool, queue_size=1, latch=True)

        # function to be exectured on shutdown
        rospy.on_shutdown(stop_listening)

        # start keyboard listener
        self.run = False
        listen_keyboard(on_press=self.key_press)

    def key_press(self, key):
        """detect keyboard commands"""

        if key == "s":
           self.run = True
        elif key == "e":
           self.run = False

        msg = Bool()
        msg.data = self.run
        self.pub.publish(msg)


def start_controller():

    # initialize the motion planner
    params = car_parameter()
    settings = parse_settings(CONTROLLER, RACETRACK, False)
    exec('controller = ' + CONTROLLER + '(params, settings)')

    # start control cycle
    PublisherSubscriber(locals()['controller'])

def start_observer():

    # initialize the observer
    params = car_parameter()
    settings = parse_settings(OBSERVER, RACETRACK, False)
    exec('observer = ' + OBSERVER + '(params, settings, x0[0], x0[1], x0[2])')

    # start control cycle
    Observer(locals()['observer'])

def start_keyboard():

    # start keyboard listener
    Keyboard()
