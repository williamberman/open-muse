#!/bin/bash
#SBATCH --job-name=muse-add-dedup-data
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=2
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/muse_add_dedup_data.txt


set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/add_dedup_data.py --slurm --start_shard 0 --end_shard 60580 --dataset laion_5
