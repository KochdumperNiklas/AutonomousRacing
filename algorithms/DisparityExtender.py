import numpy as np
import matplotlib.pyplot as plt
from auxiliary.process_lidar_data import process_lidar_data

class DisparityExtender:
    """class representing the disparity extender motion planner"""

    def __init__(self, params, settings):
        """class constructor"""

        self.params = params
        self.settings = settings

    def plan(self, x, y, theta, v, scans):
        """compute control inputs"""

        # preprocess LiDAR data
        self.radians_per_point = 0.00436332309619
        proc_ranges = self.preprocess_lidar(scans)

        # compute difference between adjacent points in the lidar data
        differences = self.get_differences(proc_ranges)

        # find disparities between adjacent points
        disparities = self.get_disparities(differences, self.settings['DIFFERENCE_THRESHOLD'])

        proc_ranges = self.extend_disparities(disparities, proc_ranges,
                                              self.params['width'], self.settings['SAFETY_PERCENTAGE'])
        steering_angle = self.get_steering_angle(proc_ranges.argmax(), len(proc_ranges))
        speed = self.settings['SPEED']

        # visualize the planned trajectory
        if self.settings['VISUALIZE']:
            self.visualization(scans, proc_ranges, proc_ranges.argmax(), speed, disparities)

        return speed, steering_angle

    def preprocess_lidar(self, ranges):
        """preprocess the lidar scan array"""

        # remove quadrant of LiDAR directly behind the car
        eighth = int(len(ranges) / 8)

        return np.array(ranges[eighth:-eighth])

    def get_differences(self, ranges):
        """compute the absolute difference between adjacent elements in the LiDAR data"""

        differences = [0.]

        for i in range(1, len(ranges)):
            differences.append(abs(ranges[i] - ranges[i - 1]))

        return differences

    def get_disparities(self, differences, threshold):
        """determine LiDAR points that were greatly different to their adjacent point"""

        disparities = []

        for index, difference in enumerate(differences):
            if difference > threshold:
                disparities.append(index)

        return disparities

    def extend_disparities(self, disparities, ranges, car_width, extra_pct):
        """for each disparity choose which side to cover"""

        width_to_cover = (car_width / 2) * (1 + extra_pct / 100)

        # loop over all disparities
        for index in disparities:

            # choose which side to cover
            first_idx = index - 1
            points = ranges[first_idx:first_idx + 2]
            close_idx = first_idx + np.argmin(points)
            far_idx = first_idx + np.argmax(points)
            cover_right = close_idx < far_idx

            # average ranges over a certain number of points
            close_dist = ranges[close_idx]
            num_points_to_cover = self.get_num_points_to_cover(close_dist, width_to_cover)
            ranges = self.cover_points(num_points_to_cover, close_idx, cover_right, ranges)

        return ranges

    def get_steering_angle(self, range_index, range_len):
        """determine steering angle for the selected point"""

        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_point
        steering_angle = np.clip(lidar_angle, np.radians(-90), np.radians(90)) * self.settings['STEER_CORRECTION']

        return steering_angle

    def get_num_points_to_cover(self, dist, width):
        """determine number of LiDAR points that correspond to the selected segment"""

        angle = 2 * np.arcsin(min(max(-1.0, width / (2 * dist)), 1.0))
        num_points = int(np.ceil(angle / self.radians_per_point))

        return num_points

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        """cover a number of LiDAR points with the distance of a closer LiDAR point to increase robustness"""

        new_dist = ranges[start_idx]

        if cover_right:
            for i in range(num_points):
                next_idx = start_idx + 1 + i
                if next_idx >= len(ranges):
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        else:
            for i in range(num_points):
                next_idx = start_idx - 1 - i
                if next_idx < 0:
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist

        return ranges

    def visualization(self, scans, proc_ranges, index, speed, disparities):
        """visualize the planned trajectory"""

        plt.cla()
        lidar_data = process_lidar_data(scans)
        plt.plot(lidar_data[0, :], lidar_data[1, :], '.r', label='lidar measurements')
        for i in range(len(disparities)):
            phi = (disparities[i] - (len(proc_ranges) / 2)) * self.radians_per_point
            x = np.array([proc_ranges[disparities[i]] * np.cos(phi), proc_ranges[disparities[i]] * np.sin(phi)])
            if i == len(disparities)-1:
                plt.plot([0, x[0]], [0, x[1]], 'b', label='disparities')
            else:
                plt.plot([0, x[0]], [0, x[1]], 'b')
        phi = (index - (len(proc_ranges) / 2)) * self.radians_per_point
        x = np.array([speed * np.cos(phi), speed * np.sin(phi)])
        plt.plot([0, x[0]], [0, x[1]], 'g', label='velocity vector')
        plt.axis('equal')
        x_min = np.min(lidar_data[0, :]) - 2
        x_max = np.max(lidar_data[0, :]) + 2
        y_min = np.min(lidar_data[1, :]) - 2
        y_max = np.max(lidar_data[1, :]) + 2
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.legend(loc='upper right')
        plt.pause(0.001)
