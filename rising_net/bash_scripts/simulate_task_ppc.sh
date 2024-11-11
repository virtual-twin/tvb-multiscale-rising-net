#!/bin/bash

CMD="python"
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="task_run_fit_plot.py"
DEF_ARGS="--FUNCMODE PPCSIM --BASENAME FIT_TASK --fitlabel allsamples"

while getopts p:s:a: flag
do
    case "${flag}" in
        p) CMD_PATH=${OPTARG};;
        s) CMD_SCRIPT=${OPTARG};;
        a) ARGS=${OPTARG};;
    esac
done

echo ${ARGS}
CMD_ARGS1="${DEF_ARGS} --MODE TVB ${ARGS}"
echo ${CMD_ARGS1}
CMD_ARGS2="${DEF_ARGS} --MODE TVB_CEREBOFF ${ARGS}"
echo ${CMD_ARGS2}

CMD1="$CMD $CMD_PATH/$CMD_SCRIPT ${CMD_ARGS1}"
echo ${CMD1}
CMD2="$CMD $CMD_PATH/$CMD_SCRIPT ${CMD_ARGS2}"
echo ${CMD2}

${CMD1}
${CMD2}