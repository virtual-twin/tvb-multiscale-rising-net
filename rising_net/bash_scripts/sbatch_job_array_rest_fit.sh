#!/bin/bash

#SBATCH --job-name=FIT_REST
#SBATCH --output=./outputs/FIT_REST/fit/logs/FIT_REST_%A_%a.out
#SBATCH --error=./outputs/FIT_REST/fit/errors/FIT_REST_%A_%a.err
#SBATCH --array=0-10
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=dionysios.perdikis@bih-charite.de

CMD_PATH="/home/docker/packages/tvb-multiscale/rising_net/bash_scripts/"
CMD="sh ${CMD_PATH}fit_rest_jobarr.sh ${SLURM_ARRAY_TASK_ID}"
apptainer exec --bind $SCRATCH:$DOCKER_SCRATCH,$RISING_NET:$DOCKER_MULTISCALE --pwd $DOCKER_SCRATCH $RISING_NET_IMAGE $CMD
