#!/bin/bash

export FASTDDS_BUILTIN_TRANSPORTS=UDPv4
export isaac_sim_package_path=$HOME/isaacsim
export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros_lib_path="$isaac_sim_package_path/exts/isaacsim.ros2.core/$ROS_DISTRO/lib"
case ":${LD_LIBRARY_PATH:-}:" in
  *":$ros_lib_path:"*) ;;
  *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ros_lib_path" ;;
esac

# Run Isaac Sim
$isaac_sim_package_path/isaac-sim.sh
