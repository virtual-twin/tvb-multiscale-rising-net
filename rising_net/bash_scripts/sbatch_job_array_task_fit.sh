#!/bin/bash

#SBATCH --job-name=FIT_TASK
#SBATCH --output=./outputs/FIT_TASK/fit/logs/TASK_%A_%a.out
#SBATCH --error=./outputs/FIT_TASK/fit/errors/TASK_%A_%a.err
#SBATCH --array=4-8
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=dionysios.perdikis@bih-charite.de

CMD_PATH="/home/docker/packages/tvb-multiscale/rising_net/bash_scripts/"
CMD="bash ${CMD_PATH}fit_task_jobarr.sh ${SLURM_ARRAY_TASK_ID}"
apptainer exec --bind $SCRATCH:$DOCKER_SCRATCH,$RISING_NET:$DOCKER_MULTISCALE --pwd $DOCKER_SCRATCH $RISING_NET_IMAGE $CMD
