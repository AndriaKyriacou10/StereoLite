#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=A100
#PBS -N finetune_frozen_fnet
#PBS -l walltime=02:00:00

# Go to the project directory where you submitted the job

cd "$PBS_O_WORKDIR"
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate py312

# --- Failsafe Checks ---
echo "Running on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
which python
# -----------------------

# --- STAGE DATASET TO LOCAL SSD ---
echo "Staging Middlebury dataset to local SSD ($TMPDIR)..."
# Copy the Middlebury folder from the network drive to the node's local SSD
cp -r ./data/datasets/Middlebury $TMPDIR/
# ----------------------------------

echo "Starting training..."
python finetune_new_loss.py

