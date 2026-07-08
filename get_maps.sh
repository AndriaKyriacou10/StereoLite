#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=A100
#PBS -N inference_job
#PBS -l walltime=00:10:00

# Go to the project directory where you submitted the job

cd "$PBS_O_WORKDIR"
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate py312

python inference.py

echo "--- Job Finished ---"