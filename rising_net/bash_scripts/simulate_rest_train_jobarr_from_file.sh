#!/bin/bash

JOBARRID=$1
CMD="python"
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="rest_run_fit_plot.py"
CMD_SCRIPTID_PATH='/home/docker/scratch'
CMD_SCRIPTID_FILE="jobarray.txt"
CMDID="simulate_rest_train_ids_args"
DEF_ARGS="--FUNCMODE TRAINSIM --MODE TVB --BASENAME FIT_REST"

$DOCKER_PYTHONPATH_CMD

ARGSIDS=$(sed -n $(($JOBARRID+1))p $CMD_SCRIPTID_PATH/$CMD_SCRIPTID_FILE)
echo ${ARGSIDS}

CMD_ARGS="${DEF_ARGS} ${ARGSIDS}"
echo ${CMD_ARGS}

CMD="$CMD $CMD_PATH/$CMD_SCRIPT ${CMD_ARGS}"
echo ${CMD}

${CMD}
