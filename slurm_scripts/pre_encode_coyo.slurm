#!/bin/bash
#SBATCH --job-name=muse-pre-encode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=8
#SBATCH --partition=dev-cluster
#SBATCH --output=/fsx/william/logs/muse_pre_encode.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/pre_encode.py --dataset coyo --start_shard 0 --end_shard 20 --slurm