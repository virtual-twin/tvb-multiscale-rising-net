#!/bin/bash

#SBATCH --job-name=FIT_REST
#SBATCH --output=./outputs/FIT_REST/logs/%A_%a.out
#SBATCH --error=./outputs/FIT_REST/errors/%A_%a.out
#SBATCH --array=0-32999%33
#SBATCH --time=10:00
#SBATCH --mem=4G

