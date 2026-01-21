#!/bin/bash

#SBATCH --job-name=FTmn
#SBATCH --output=./outputs/FIT_TASK/mean_sims/allsamples/logs/FIT_TASK_MEAN_%A_%a.out
#SBATCH --error=./outputs/FIT_TASK/mean_sims/allsamples/errors/FIT_TASK_MEAN_%A_%a.err
#SBATCH --array=0-49
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=dionysios.perdikis@bih-charite.de

JOB_ID=$((${SLURM_ARRAY_TASK_ID}))
CMD_PATH="/home/docker/packages/tvb-multiscale/rising_net/bash_scripts/"
CMD="bash ${CMD_PATH}simulate_task_mean_jobarr.sh ${JOB_ID}"
apptainer exec --bind $SCRATCH:$DOCKER_SCRATCH,$RISING_NET:$DOCKER_MULTISCALE --pwd $DOCKER_SCRATCH $RISING_NET_SANDBOX $CMD
