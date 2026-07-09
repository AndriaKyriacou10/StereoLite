#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=A100
#PBS -N inference_CV
#PBS -l walltime=00:05:00

# Go to the project directory where you submitted the job

cd "$PBS_O_WORKDIR"
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate py312

python inference.py -dataset eth3d --seed 42 --compute_cost_volume --ckpt ./checkpoints/phase1_best_model.pth --out_dir ./inference_images/with_cv


echo "--- Job Finished ---"