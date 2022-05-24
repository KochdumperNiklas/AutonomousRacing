import numpy as np
import matplotlib.pyplot as plt
from auxiliary.process_lidar_data import process_lidar_data
from auxiliary.adaptive_cruise_control import adaptive_cruise_control
from auxiliary.FakeMapSimulator import FakeMapSimulator

class GapFollower:
    """class representing the gap follower motion planner"""

    def __init__(self, params, settings):
        """class constructor"""

        # store algorithm settings and car parameter
        settings['STRAIGHTS_STEERING_ANGLE'] = np.deg2rad(settings['STRAIGHTS_STEERING_ANGLE'])
        self.settings = settings
        self.params = params
        self.radians_per_elem = None

        # initialize fake map for lidar scan simulation
        if settings['FAKE_MAP'] == "":
            self.fake_map_simulator = None
        else:
            self.fake_map_simulator = FakeMapSimulator(settings['FAKE_MAP'])

    def plan(self, x, y, theta, v, scans):
        """compute control inputs"""

        # potentially get LiDAR scan from fake map
        if not self.fake_map_simulator is None:
            tmp = self.fake_map_simulator.get_scan(x, y, theta, scans)
            if max(tmp) > 0:
                scans = tmp

        # eliminate all points inside 'bubble' (set them to zero)
        proc_ranges = self.preprocess_lidar(scans)
        closest = proc_ranges.argmin()
        min_index = closest - self.settings['BUBBLE_RADIUS']
        max_index = closest + self.settings['BUBBLE_RADIUS']
        if min_index < 0:
            min_index = 0
        if max_index >= len(proc_ranges):
            max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # select speed
        steering_angle = self.get_angle(best, len(proc_ranges))

        if abs(steering_angle) > self.settings['STRAIGHTS_STEERING_ANGLE']:
            speed = self.settings['CORNERS_SPEED']
        else:
            if self.detect_straight(scans, self.params['width']):
                speed = self.settings['LONG_STRAIGHTS_SPEED']
            else:
                speed = self.settings['STRAIGHTS_SPEED']

        if self.settings['ADAPTIVE_CRUISE_CONTROL']:
            speed = adaptive_cruise_control(scans, speed, self.params['width'])

        # visualized the planned trajectory
        if self.settings['VISUALIZE']:
            self.visualization(scans, gap_start, gap_end, best, speed)

        return np.array([speed, steering_angle])

    def preprocess_lidar(self, ranges):
        """preprocess the lidar scan array"""

        self.radians_per_elem = 0.00436332309619

        # don't use the lidar data from directly behind the car
        proc_ranges = np.array(ranges[self.settings['LIDAR_RANGE']:-self.settings['LIDAR_RANGE']])

        # sets each value to the mean over a given window
        proc_ranges = np.convolve(proc_ranges,
                    np.ones(self.settings['PREPROCESS_CONV_SIZE']), 'same') / self.settings['PREPROCESS_CONV_SIZE']

        # reject large values
        max_range = False

        for i in range(len(proc_ranges)):
            if proc_ranges[i] >= self.settings['MAX_LIDAR_DIST']:
                if not max_range:
                    start = i
                    max_range = True
            elif max_range:
                end = i
                proc_ranges[start:end] = proc_ranges[end]
                max_range = False

        if max_range:
            proc_ranges[start:] = proc_ranges[start-1]

        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """return the start and end index of the max gap"""

        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)

        # get a slice for each contiguous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]

        # determine maximum slice
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl

        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        """find the best point in the gap"""

        # sliding window average over the data in the max gap (will help the car to avoid hitting corners)
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.settings['BEST_POINT_CONV_SIZE']),
                                       'same') / self.settings['BEST_POINT_CONV_SIZE']

        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """determine steering angle for the selected point"""

        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2

        return steering_angle

    def detect_straight(self, scans, width):
        """detect if the car is at the long straight and increase the speed if it is"""

        # reject large values
        proc_ranges = scans
        max_range = False

        for i in range(len(proc_ranges)):
            if proc_ranges[i] >= self.settings['MAX_LIDAR_DIST']:
                if not max_range:
                    start = i
                    max_range = True
            elif max_range:
                end = i
                proc_ranges[start:end] = proc_ranges[end]
                max_range = False

        if max_range:
            proc_ranges[start:] = proc_ranges[start - 1]

        # settings
        angle = 0

        # transform LiDAR data to a point cloud
        lidar_data = process_lidar_data(proc_ranges)

        # determine points that are in the path of the vehicle
        tmp = np.dot(np.array([[-np.sin(angle), np.cos(angle)]]), lidar_data)
        ind = np.where(abs(tmp[0]) < width)[0]

        # determine minimum distance to the vehicle in front
        tmp = np.dot(np.array([[np.cos(angle), np.sin(angle)]]), lidar_data[:, ind])[0]
        index = np.where(tmp > 0)
        dist = np.min(tmp[index[0]])

        return dist > 4

    def visualization(self, scans, gap_start, gap_end, best, speed):
        """visualize the planned trajectory"""

        plt.cla()
        lidar_data = process_lidar_data(scans)
        ind = np.where(scans < self.settings['MAX_LIDAR_DIST'])
        lidar_data = lidar_data[:, ind[0]]
        proc_ranges = scans[self.settings['LIDAR_RANGE']:-self.settings['LIDAR_RANGE']]
        plt.plot(lidar_data[0, :], lidar_data[1, :], '.r', label='lidar measurements')
        phi1 = 2*self.get_angle(gap_start, len(proc_ranges))
        phi2 = 2*self.get_angle(gap_end, len(proc_ranges))
        phi3 = 2*self.get_angle(best, len(proc_ranges))
        x1 = np.array([proc_ranges[gap_start] * np.cos(phi1), proc_ranges[gap_start] * np.sin(phi1)])
        x2 = np.array([proc_ranges[gap_end-1] * np.cos(phi2), proc_ranges[gap_end-1] * np.sin(phi2)])
        x3 = np.array([speed * np.cos(phi3), speed * np.sin(phi3)])
        plt.plot([0, x1[0], x2[0], 0], [0, x1[1], x2[1], 0], 'b', label='largest gap')
        plt.plot([0, x3[0]], [0, x3[1]], 'g', label='velocity vector')
        plt.axis('equal')
        x_min = np.min(lidar_data[0, :]) - 2
        x_max = np.max(lidar_data[0, :]) + 2
        y_min = np.min(lidar_data[1, :]) - 2
        y_max = np.max(lidar_data[1, :]) + 2
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.legend(loc='upper right')
        plt.pause(0.001)
