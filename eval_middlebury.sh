#!/bin/bash
#PBS -l walltime=00:05:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=A100
#PBS -N eval_mid_no_cv

cd "$PBS_O_WORKDIR"
eval "$(~/miniforge3/bin/conda shell.bash hook)"

conda activate py312

python evaluate_other_datasets.py --dataset middlebury --ckpt ./checkpoints/phase2_best_model.pth

echo "--- Job Finished ---"