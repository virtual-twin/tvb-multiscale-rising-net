#!/bin/bash

JOBARRID=$1
CMD="python"
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="task_run_fit_plot.py"
CMD_SCRIPTID="bash_utils.py"
CMDID="simulate_task_mapmean_ids_args"
BASIC_ARGS="--FUNCMODE MEANSIM --BASENAME FIT_TASK --fitlabel allsamples"
DEF_ARGS="$BASIC_ARGS --REST_BASENAME FIT_REST --restfitlabel allsamples"

$DOCKER_PYTHONPATH_CMD

CMDIDS="$CMD $CMD_PATH/$CMD_SCRIPTID $CMDID $JOBARRID"
echo ${CMDIDS}

ARGSIDS=$($CMDIDS)
echo ${ARGSIDS}

ARGS1="${DEF_ARGS} --MODE TVB"
echo ${ARGS1}
ARGS2="${DEF_ARGS} --MODE TVB_CEREBOFF"
echo ${ARGS2}

# for iR in {0..2}; do

CMD1="$CMD $CMD_PATH/$CMD_SCRIPT ${ARGS1} ${ARGSIDS}"
echo ${CMD1}
${CMD1}

CMD2="$CMD $CMD_PATH/$CMD_SCRIPT ${ARGS2} ${ARGSIDS}"
echo ${CMD2}
${CMD2}

# done

#PLOTCMD="$CMD $CMD_PATH/$CMD_SCRIPT --function load_and_plot_comparisons $BASIC_ARGS"
#echo $PLOTCMD
#$PLOTCMD
