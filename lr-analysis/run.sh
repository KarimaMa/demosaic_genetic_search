#!/bin/bash

ROOT="/home/karima/demosaic_genetic_search"

cd $ROOT
SAVE=$1
MODELS=$2
GPUS=$3
BATCHSIZE=$4

python lr-analysis/run.py \
  --save=${SAVE} \
  --modeldir=${MODELS} \
  --num_gpus=${GPUS} \
  --batch_size=${BATCHSIZE} \
  --epochs=1 \
  --training_file=../cnn-data/50k-train-files.txt \
  --validation_file=../cnn-data/val_files.txt \
  --green_model_asts=PARETO_GREEN_MODELS/flagship-green-pareto-files/pareto_green_asts.txt \
  --green_model_weights=PARETO_GREEN_MODELS/flagship-green-pareto-files/pareto_green_weights.txt \
  --sample_n=4 \
  --full_model
