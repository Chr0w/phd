#!/bin/bash

echo "Starting image $1 with container name $2 ..."

xhost local:root
XAUTH=/tmp/.docker.xauth

echo "Removing target container name if already in use...."
docker rm $2

docker run -it \
    --name=$2 \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --net=host \
    --privileged \
    $1 \
    bash



