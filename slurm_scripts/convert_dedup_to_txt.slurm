#!/bin/bash
#SBATCH --job-name=muse-convert-dedup-data-to-txt
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/muse_convert_dedup_data_to_txt.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/convert_dedup_to_txt.py --slurm