#!/bin/bash
cat << 'EOF' >> ~/.bashrc

export _colcon_cd_root=~/ros2_ws

# ----- Git -----
alias gs='git status'

function git_upload() {
   echo "Adding, commiting and pushing all changes"
   git add --all && git commit -m "$1" && git push
}

# ----- Bash -----
alias editbash='gedit ~/.bashrc'

# ----- Docker -----
alias docker_stop_all_containers='docker kill $(sudo docker ps -a)'
alias docker_remove_all_containers='docker rm $(docker ps -a -q)'
alias docker_remove_all_images='docker rmi $(docker images  -q)'
alias docker_clear_all='docker_remove_all_containers && docker_remove_all_images'

# ----- ROS2 -----
alias sr2='source /opt/ros/$ROS_DISTRO/setup.bash'
alias rviz='ros2 run rviz2 rviz2'
#export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ----- Project -----
# Start Isaac Sim
sis () {
cd ~/phd && ./start_isaac_sim.sh
}

# Start system
sys () {
cd ~/phd/ros2_ws
source install/setup.bash
cd launch
ros2 launch system_launch.py
}

# ros build (colcon)
rbld() {
sr2 && colcon build
}

# Start Foxglove
foxglove () {
sr2 && ros2 launch rosbridge_server rosbridge_websocket_launch.xml
}

EOF
