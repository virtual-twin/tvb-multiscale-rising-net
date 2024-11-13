#!/bin/bash

JOBARRID=$1
CMD="python"
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="task_run_fit_plot.py"
CMD_SCRIPTID="bash_utils.py"
CMDID="simulate_task_train_ids_args"
DEF_ARGS="--FUNCMODE TRAINSIM --BASENAME FIT_TASK"

$DOCKER_PYTHONPATH_CMD

CMDIDS="$CMD $CMD_PATH/$CMD_SCRIPTID $CMDID $JOBARRID"
echo ${CMDIDS}

ARGSIDS=$($CMDIDS)
echo ${ARGSIDS}

CMD_ARGS1="${DEF_ARGS} ${ARGSIDS} --MODE TVB"
echo ${CMD_ARGS1}
CMD_ARGS2="${DEF_ARGS} ${ARGSIDS} --MODE TVB_CEREBOFF"
echo ${CMD_ARGS2}

CMD1="$CMD $CMD_PATH/$CMD_SCRIPT ${CMD_ARGS1}"
echo ${CMD1}
${CMD1}

CMD2="$CMD $CMD_PATH/$CMD_SCRIPT ${CMD_ARGS2}"
echo ${CMD2}
${CMD2}