#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=A100
#PBS -N eval_no_cv_mid
#PBS -l walltime=00:05:00

# Go to the project directory where you submitted the job

cd "$PBS_O_WORKDIR"
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate py312

python evaluate_other_datasets.py --dataset middlebury --ckpt ./checkpoints/phase1_best_model.pth --cost_volume

echo "--- Job Finished ---"