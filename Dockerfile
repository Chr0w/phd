FROM osrf/ros:humble-desktop-full

RUN apt-get update
RUN . opt/ros/humble/setup.sh

RUN echo "All done!"