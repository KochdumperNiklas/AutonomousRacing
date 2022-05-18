import os.path
import warnings
import numpy as np
import pandas as pd
from auxiliary.ScanSimulator import ScanSimulator
from auxiliary.Polygon import Polygon

class FakeMapSimulator:
    """class for simulating LiDAR scans using a fake map"""

    def __init__(self, fake_map):
        """class constructor"""

        # load the fake map
        self.scanner, self.positions = self.load_fake_map(fake_map)

    def get_scan(self, x, y, theta, scans):
        """return the LiDAR scan for the current position using the fake scan"""

        # check if fake map is active
        p = np.array([[x], [y]])
        active = False

        for pos in self.positions:
            if pos.contains(p):
                active = True

        # generate the fake scan
        if active:
            scans = self.scanner.scan(np.array([x, y, theta]))

        return scans

    def load_fake_map(self, fake_map):
        """load the fake map that is used to simulate lidar scans"""

        # initialize file paths
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        map_path = os.path.join(parent, 'racetracks', fake_map, fake_map + '.png')
        yaml_path = os.path.join(parent, 'racetracks', fake_map, fake_map + '.yaml')
        pos_path = os.path.join(parent, 'racetracks', fake_map, 'fake_map_positions.csv')

        # check if map and required files exist
        if not os.path.exists(map_path):
            raise Exception('The specified fake map does not exist!')

        if not os.path.exists(yaml_path):
            raise Exception('The configuration file for the specified fake map is missing')

        if not os.path.exists(pos_path):
            msg = 'Could not find file /racetracks/' + fake_map + '/fake_map_positions.csv that'
            msg = msg + ' specifies the positions where the fake map is used. Using fake map everywhere.'
            warnings.warn(msg)

        # initialize scan simulator
        scanner = ScanSimulator(map_path, yaml_path)

        # load fake map positions
        if not os.path.exists(pos_path):
            positions = None
        else:
            positions = self.load_positions(pos_path)

        return scanner, positions

    def load_positions(self, path):
        """load the regions where the fake map is active"""

        # load data from .csv file
        data = np.asarray(pd.read_csv(path, header=None))

        if not np.mod(data.shape[1], 2) == 0:
            raise Exception('File ' + path + ' has a wrong format!')

        # loop over all regions
        positions = []

        for i in range(0, data.shape[1], 2):
            ind = np.where(~np.isnan(data[:, i]))
            x = data[ind[0], i]
            y = data[ind[0], i+1]
            positions.append(Polygon(x, y))

        return positions
