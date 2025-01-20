#!/bin/bash

#SBATCH --job-name=FTPPCf
#SBATCH --output=./outputs/FIT_TASK/PPC_sims/logs/FTPPCf_%A_%a.out
#SBATCH --error=./outputs/FIT_TASK/PPC_sims/errors/FTPPCf_%A_%a.err
#SBATCH --array=1-1709
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --mail-user=dionysios.perdikis@bih-charite.de

JOB_ID=$((${SLURM_ARRAY_TASK_ID}))
CMD_PATH="/home/docker/packages/tvb-multiscale/rising_net/bash_scripts/"
CMD="bash ${CMD_PATH}simulate_task_ppc_jobarr_from_file.sh ${JOB_ID}"
apptainer exec --bind $SCRATCH:$DOCKER_SCRATCH,$RISING_NET:$DOCKER_MULTISCALE --pwd $DOCKER_SCRATCH $RISING_NET_IMAGE $CMD
