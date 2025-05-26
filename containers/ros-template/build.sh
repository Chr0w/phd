#!/bin/bash

export BUILDKIT_PROGRESS=plain

#adduser -u ${UID} --disabled-password --gecos "" appuser

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker compose -f $SCRIPT_DIR/docker-compose.yml build 
