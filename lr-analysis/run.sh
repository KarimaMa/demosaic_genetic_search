#!/bin/bash

HOME2=/nobackup/users/karima
PYTHON_VIRTUAL_ENVIRONMENT=python3.8
CONDA_ROOT=$HOME2/anaconda3
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

cd $HOME2/demosaic_genetic_search

SAVE=$1
MODELS=$2
GPUS=$3
BATCHSIZE=$4

python lr-analysis/run.py \
  --save=${SAVE} \
  --modeldir=${MODELS} \
  --num_gpus=${GPUS} \
  --batch_size=${BATCHSIZE} \
  --epochs=6 \
  --training_file=/nobackup/users/karima/subset7_pkg/subset7_100k_train_files.txt \
  --validation_file=/nobackup/users/karima/subset7_pkg/subset7_100k_val_files.txt \
  --green_model_asts=PARETO_GREEN_MODELS/flagship-green-pareto-files/pareto_green_asts.txt \
  --green_model_weights=PARETO_GREEN_MODELS/flagship-green-pareto-files/pareto_green_weights.txt \
  --sample_n=500 \
  --full_model
