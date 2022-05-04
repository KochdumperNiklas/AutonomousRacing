# Autonomous Racing

This repository contains control algorithms for autonomous racing with a F1tenth racecar. All algorithms can either be used for 
simulation using the file *main_gym.py*, or for controlling the real pyhsical car using the file *main_car.py*. The desired 
algorithm and racetrack can be specified directly in these two main files. Fine-tuned algorithm settings for a specific 
racetrack can be specified in the corresponding config-files *CONTROLLER.cfg* in the directory */racetracks/RACETRACK/settings/*. 
The simulation environment also supports head-to-head racing of two racecars.

## Installation

For the simulation environment it is required that the F1tenth gym https://github.com/f1tenth/f1tenth_gym is installed.

## Adding New Controllers

Every controller should be implemented as an own class with constructor syntax

`__init__(self, params, settings)`

where `params` is a dictionary containing the vehicle parameter (widht, length, mass, etc.), and `settings` is a dictionary 
containing the parsed algorithm settings read from the corresponding *CONTROLLER.cfg* file. A config-file that defines valid values 
for the algorithm has to be provided in the directory *settings*. Moreover, each controller class has to implement the function 

`plan(self, x, y, theta, v, scans)`

which gets the positions `x` and `y`, the orientation `theta`, and the velocity `v` of the car as well as the current LiDAR 
measurements `scans`, and returns the control commands `speed` and `steer`. All controllers should be implemented as seperate files 
in the directory */algorithms/*.

## Adding New Racetracks

For each racetrack a file *RACETRACK.png* that stores an image of the racetrack, a file *RACETRACK.yaml* that defines the origin of 
the racetrack, and a file *config_RACETRACK.yaml* that defines the start position of the racecar on the racetrack are required. All 
racetracks have to be contained in the directory */racetracks/*.  
