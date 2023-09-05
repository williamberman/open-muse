#!/bin/bash
#SBATCH --job-name=muse-stability-metadata-annotate-laion-475
#SBATCH --nodes=10
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --partition=production-cluster
#SBATCH --output=/fsx/william/logs/muse_stability_metadata_annotate_laion_475_resized_to_max_1024.txt

set -e -u

source /fsx/william/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /fsx/william/open-muse

time srun python scripts/stability_metadata_annotate/stability_metadata_annotate.py --slurm --start_shard 0 --end_shard 58886 --update_orig_dataset laion_475_resized_to_max_1024