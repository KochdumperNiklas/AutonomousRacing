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
