#!/bin/bash

JOBARRID=$1
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="bash_utils.py"
CMDFUN="simulate_cosim_lesion_nest"

$DOCKER_PYTHONPATH_CMD

CMD="python $CMD_PATH/$CMD_SCRIPT $CMDFUN $JOBARRID"
echo ${CMD}
${CMD}
