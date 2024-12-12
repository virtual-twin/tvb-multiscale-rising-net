#!/bin/bash

#SBATCH --job-name=FIT_REST0
#SBATCH --output=./outputs/FIT_REST/logs/FIT_REST0_%A_%a.out
#SBATCH --error=./outputs/FIT_REST/errors/FIT_REST0_%A_%a.err
#SBATCH --array=0-5499%100
#SBATCH --time=30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=dionysios.perdikis@bih-charite.de

JOB_ID=$((${SLURM_ARRAY_TASK_ID} + 0*5500))
CMD_PATH="/home/docker/packages/tvb-multiscale/rising_net/bash_scripts/"
CMD="bash ${CMD_PATH}simulate_rest_train_jobarr.sh ${JOB_ID}"
apptainer exec --bind $SCRATCH:$DOCKER_SCRATCH,$RISING_NET:$DOCKER_MULTISCALE --pwd $DOCKER_SCRATCH $RISING_NET_IMAGE $CMD
