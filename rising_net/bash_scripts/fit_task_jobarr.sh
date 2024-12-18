#!/bin/bash

JOBARRID=$1
CMD="python"
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="bash_utils.py"
PYTHON_CMD="fit_task"

$DOCKER_PYTHONPATH_CMD

CMD="$CMD $CMD_PATH/$CMD_SCRIPT $PYTHON_CMD $JOBARRID"
echo ${CMD}

${CMD}
