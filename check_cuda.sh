#!/bin/bash
#PBS -lselect=1:ncpus=4:mem=10gb:ngpus=1
#PBS -lwalltime=1:0:0
#PBS -N check_cuda
cd $PBS_O_WORKDIR

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate py312

## Verify install:
python -c "import torch;print(torch.cuda.is_available())"
