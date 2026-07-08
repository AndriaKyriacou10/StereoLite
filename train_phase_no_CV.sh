#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=A100
#PBS -N training_phase2
#PBS -l walltime=15:00:00

# Go to the project directory where you submitted the job

cd "$PBS_O_WORKDIR"
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate py312

# --- Failsafe Checks ---
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
which python
# -----------------------

python train_stereo.py --epochs 50

echo "Training job finished successfully!"