#!/bin/bash
#SBATCH --job-name=download-midjourney-v5-202304-clean-upscaled
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/download_midjourney_v5_202304_clean_upscaled.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse.2

time srun python scripts/download_midjourney_v5_202304_clean.py --slurm --upscaled_dataset