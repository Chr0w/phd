Sourcing this env:
source install/setup.bash

To set logging directory:
export ROS_LOG_DIR=~/phd/ros2_ws/log

Use:
ros2 run rqt_console rqt_console
To start up a GUI logging tool 

To get info on some ROS2 message type:
ros2 interface show sensor_msgs/msg/LaserScan

