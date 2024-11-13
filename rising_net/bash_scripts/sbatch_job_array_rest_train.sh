#!/bin/bash

#SBATCH --job-name=FIT_REST
#SBATCH --output=./outputs/FIT_REST/logs/FIT_REST_%A_%a.out
#SBATCH --error=./outputs/FIT_REST/errors/FIT_REST_%A_%a.out
#SBATCH --array=0-32999%330
#SBATCH --time=12:00
#SBATCH --mem=2G

CMD="sh simulate_rest_train_jobarr.sh ${SLURM_ARRAY_TASK_ID}"
apptainer exec --bind $SCRATCH:$DOCKER_SCRATCH,$RISING_NET:$DOCKER_MULTISCALE --pwd $DOCKER_SCRATCH $RISING_NET_IMAGE $CMD
