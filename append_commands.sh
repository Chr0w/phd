#!/bin/bash
cat << 'EOF' >> ~/.bashrc

export _colcon_cd_root=~/ros2_ws

# Git
alias gs='git status'

function git_upload() {
   echo "Adding, commiting and pushing all changes"
   git add . && git commit -m "$1" && git push
}

# Docker
alias docker_stop_all_containers='docker kill $(sudo docker ps -a)'
alias docker_remove_all_containers='docker rm $(docker ps -a -q)'
alias docker_remove_all_images='docker rmi $(docker images  -q)'
alias docker_clear_all='docker_remove_all_containers && docker_remove_all_images'

alias sr2='source /opt/ros/humble/setup.bash'
alias rviz='ros2 run rviz2 rviz2'

# Project

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

# Start Foxglove
foxglove () {
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
}

EOF
