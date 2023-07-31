#!/bin/bash
#SBATCH --job-name=muse-download-dedup-idx
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/muse_download_dedup_idx.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/download_dedup_idx.py --slurm