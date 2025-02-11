#!/bin/bash

JOBARRID=$1
CMD="python"
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="rest_run_fit_plot.py"
CMD_SCRIPTID="bash_utils.py"
CMDID="simulate_rest_ppc_ids_args"
DEF_ARGS="--FUNCMODE PPCSIM --MODE TVB --BASENAME FIT_REST --fitlabel allruns"

$DOCKER_PYTHONPATH_CMD

CMDIDS="$CMD $CMD_PATH/$CMD_SCRIPTID $CMDID $JOBARRID"
echo ${CMDIDS}

ARGSIDS=$($CMDIDS)
echo ${ARGSIDS}

CMD_ARGS="${DEF_ARGS} ${ARGSIDS}"
echo ${CMD_ARGS}

CMD="$CMD $CMD_PATH/$CMD_SCRIPT ${CMD_ARGS}"
echo ${CMD}

${CMD}
