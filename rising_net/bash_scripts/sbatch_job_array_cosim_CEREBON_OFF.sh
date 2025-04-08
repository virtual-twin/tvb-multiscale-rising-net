#!/bin/bash

#SBATCH --job-name=FTcosim
#SBATCH --output=./outputs/CEREBON_OFF/logs/FTcosim_%A_%a.out
#SBATCH --error=./outputs/CEREBON_OFF/errors/FTcosim_%A_%a.err
#SBATCH --array=0-99
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=dionysios.perdikis@bih-charite.de

JOB_ID=$((${SLURM_ARRAY_TASK_ID}))
CMD_PATH="/home/docker/packages/tvb-multiscale/rising_net/bash_scripts/"
CMD="bash ${CMD_PATH}simulate_cosim_CEREBON_OFF_jobarr.sh ${JOB_ID}"
# apptainer exec --bind $SCRATCH:$DOCKER_SCRATCH,$RISING_NET:$DOCKER_MULTISCALE --pwd $DOCKER_SCRATCH $RISING_NET_IMAGE $CMD
apptainer exec --bind $SCRATCH:$DOCKER_SCRATCH,$RISING_NET:$DOCKER_MULTISCALE --pwd $DOCKER_SCRATCH $RISING_NET_SANDBOX $CMD