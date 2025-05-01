#!/bin/bash

image_name=ros_humble_img
container_name=ros_humble_container

echo "Building $image_name ..." 

docker build -t $image_name .


bash run_docker.bash "$image_name" "$container_name"