#!/bin/bash

#SBATCH --job-name=FIT_REST_MAP
#SBATCH --output=./outputs/FIT_REST/MAP_sims/logs/FIT_REST_MAP_%A_%a.out
#SBATCH --error=./outputs/FIT_REST/MAP_sims/errors/FIT_REST_MAP_%A_%a.err
#SBATCH --array=0-32
#SBATCH --time=30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=dionysios.perdikis@bih-charite.de

JOB_ID=$((${SLURM_ARRAY_TASK_ID}))
CMD_PATH="/home/docker/packages/tvb-multiscale/rising_net/bash_scripts/"
CMD="sh ${CMD_PATH}simulate_rest_map_jobarr.sh ${JOB_ID}"
apptainer exec --bind $SCRATCH:$DOCKER_SCRATCH,$RISING_NET:$DOCKER_MULTISCALE --pwd $DOCKER_SCRATCH $RISING_NET_IMAGE $CMD
