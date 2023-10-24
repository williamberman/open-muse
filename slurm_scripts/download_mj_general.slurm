#!/bin/bash
#SBATCH --job-name=download-mj-general
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/download_mj_general.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse.2

time srun python scripts/download_mj_general.py --slurm