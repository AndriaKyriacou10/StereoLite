#!/bin/bash

#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=A100
#PBS -N eval_eth

cd "$PBS_O_WORKDIR"
eval "$(~/miniforge3/bin/conda shell.bash hook)"

conda activate py312

python evaluate_stereo.py --restore_ckpt ./checkpoints/LiteAnyStereo.pth --dataset eth3d
