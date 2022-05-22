import numpy as np
import matplotlib.pyplot as plt
from auxiliary.process_lidar_data import process_lidar_data

def adaptive_cruise_control(scans, speed, width):
    """adapt the speed to keep a safe distance to the vehicle in front"""

    # settings
    angle = 0
    min_dist = 1

    # transform LiDAR data to a point cloud
    lidar_data = process_lidar_data(scans)

    # determine points that are in the path of the vehicle
    tmp = np.dot(np.array([[-np.sin(angle), np.cos(angle)]]), lidar_data)
    ind = np.where(abs(tmp[0]) < width/2)[0]

    # determine minimum distance to the vehicle in front
    tmp = np.dot(np.array([[np.cos(angle), np.sin(angle)]]), lidar_data[:, ind])[0]
    index = np.where(tmp > 0)
    dist = np.min(tmp[index[0]])

    # limit speed to keep a safe distance to the vehicle in front
    if dist < min_dist:
        speed = min(dist, speed)

    return speed
