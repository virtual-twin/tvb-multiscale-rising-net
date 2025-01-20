#!/bin/bash

JOBARRID=$1
CMD="python"
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="task_run_fit_plot.py"
CMD_SCRIPTID_PATH='/home/docker/scratch'
CMD_SCRIPTID_FILE="jobarray.txt"
BASIC_ARGS="--FUNCMODE PPCSIM --BASENAME FIT_TASK --fitlabel allsamples"
DEF_ARGS="$BASIC_ARGS --REST_BASENAME FIT_REST --restfitlabel allsamples"

$DOCKER_PYTHONPATH_CMD

ARGSIDS=$(sed -n $(($JOBARRID))p $CMD_SCRIPTID_PATH/$CMD_SCRIPTID_FILE)
echo ${ARGSIDS}

CMD_ARGS="${DEF_ARGS} ${BASIC_ARGS}  ${ARGSIDS}"
echo ${CMD_ARGS}

CMD="$CMD $CMD_PATH/$CMD_SCRIPT ${CMD_ARGS}"
echo ${CMD}
${CMD}
