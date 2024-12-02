#!/bin/bash

JOBARRID=$1
CMD="python"
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="rest_run_fit_plot.py"
DEF_ARGS="--FUNCMODE BOLDSIM --MODE TVB --BASENAME FIT_REST --fitlabel allsamples"

$DOCKER_PYTHONPATH_CMD

ARGSIDS="--iG $JOBARRID"
echo ${ARGSIDS}

CMD_ARGS="${DEF_ARGS} ${ARGSIDS}"
echo ${CMD_ARGS}

CMD="$CMD $CMD_PATH/$CMD_SCRIPT ${CMD_ARGS}"
echo ${CMD}

${CMD}
