#!/bin/bash

#SBATCH --job-name=wildcats
#SBATCH --partition=compute
#SBATCH --array=1-9999
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem=28G
#SBATCH --account=BISC019342
#SBATCH --output ./output/slurm/slurm-%A_%a.out

cd "${SLURM_SUBMIT_DIR}"

echo Time is "$(date)"

module load lang/python/miniconda/3.9.7
. ~/.bashrc

conda activate wildcats
python run_sim_sequential.py
conda deactivate

echo Time is "$(date)"
