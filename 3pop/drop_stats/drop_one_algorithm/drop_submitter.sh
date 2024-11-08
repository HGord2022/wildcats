#!/bin/bash

#SBATCH --job-name=drop
#SBATCH --partition=compute
#SBATCH --array=1-101
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --mem=10G
#SBATCH --account=BISC019342
#SBATCH --output ./slurm/slurm-%A_%a.out


cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"

module load lang/python/miniconda/3.9.7
. ~/.bashrc

conda activate bp1_envA
python drop_flow.py
conda deactivate

echo Time is "$(date)"
