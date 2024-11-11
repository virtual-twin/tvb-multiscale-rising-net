#!/bin/bash

CMD="python"
CMD_PATH='/home/docker/packages/tvb-multiscale/rising_net/scripts'
CMD_SCRIPT="task_run_fit_plot.py"
DEF_ARGS=""

while getopts p:s:a: flag
do
    case "${flag}" in
        p) CMD_PATH=${OPTARG};;
        s) CMD_SCRIPT=${OPTARG};;
        a) ARGS=${OPTARG};;
    esac
done

echo ${ARGS}
CMD_ARGS="${DEF_ARGS} ${ARGS}"
echo ${CMD_ARGS}
CMD="$CMD $CMD_PATH/$CMD_SCRIPT ${CMD_ARGS}"

echo ${CMD}
${CMD}