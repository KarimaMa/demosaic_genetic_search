#!/bin/bash

python sys_run/retrain_one_model.py \
	--model_info_dir=$1 \
	--save=$2 \
	--model_retrain_list=$3 \
	--gpu_id=$4 \
	--task_id=$5 \
	--epochs=$6 \
	--learning_rate=0.003 \
	--training_file=/home/karima/cnn-data/subset7_100k_train_files.txt \
	--validation_file=/home/karima/cnn-data/subset7_100k_val_files.txt \
