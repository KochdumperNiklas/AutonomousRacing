import numpy as np

def process_lidar_data(data):
    """convert the lidar measurement to a 2-dimenaional point cloud"""

    # parameter (defined by the lidar that is used)
    phi = -2.35619449615                                # initial angle
    delta_phi = 0.00436332309619                        # angle increment

    # loop over all lidar beams
    points = np.zeros((2, len(data)))

    for i in range(0, len(data)):
        points[0, i] = data[i] * np.cos(phi)
        points[1, i] = data[i] * np.sin(phi)
        phi = phi + delta_phi

    return points

def smooth_lidar(ranges, MAX_LIDAR_DIST, PREPROCESS_CONV_SIZE):
    """smooth out the lidar scan"""

    # don't use the LiDAR data from directly behind the car
    proc_ranges = np.array(ranges[135:-135])

    # sets each value to the mean over a given window
    proc_ranges = np.convolve(proc_ranges, np.ones(PREPROCESS_CONV_SIZE), 'same') / PREPROCESS_CONV_SIZE
    proc_ranges = np.clip(proc_ranges, 0, MAX_LIDAR_DIST)

    return proc_ranges