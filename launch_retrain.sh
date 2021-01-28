#!/bin/bash
NUM_MODELS=$1
num_gpus=$2

for i in $(seq 0 $NUM_MODELS); do  
	python sys_run/retrain_one_model.py \
		--task_id=$i \
		--gpu_id=$(($i%$num_gpus)) \
		--model_info_dir=$3 \
		--save=$4 \
		--model_retrain_list=$5 \
		--epochs=$6 \
		--learning_rate=0.003 \
		--training_file=/home/karima/cnn-data/subset7_100k_train_files.txt \
		--validation_file=/home/karima/cnn-data/subset7_100k_val_files.txt & 

	pids[(($i%4))]=$!
	if ! (( (i+1)%4)); then
		echo "all gpus loaded waiting for training jobs to finish"
		for pid in ${pids[*]}; do
			wait $pid
		done
	fi
 	unset pids
done
