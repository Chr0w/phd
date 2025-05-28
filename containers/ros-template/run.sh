#!/bin/bash

REPOSITORY_NAME="$(basename "$(dirname -- "$( readlink -f -- "$0"; )")")"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export HOST_UID=$(id -u)

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

docker compose -f $SCRIPT_DIR/docker-compose.yml run \
${REPOSITORY_NAME} bash
