#!/bin/bash
n_subsets=$1
subset_dir=$2
train_portion=$3
save_dir=$4
use_multires=$5
epochs=$6

echo $use_multires

start_gpu=1
end_gpu=3
subset_id=0

while [ $subset_id -lt $n_subsets ]; do
  for gpu_id in $(seq $start_gpu $end_gpu); do
    cmd="python3 train_eval_scripts/parallel_rand_subset_train.py"
    if [ $use_multires -eq 1 ]; then
      cmd=$cmd" --multires_model"
    fi
    cmd=$cmd" --gpu=$gpu_id" 
    cmd=$cmd" --epochs=$epochs"
    cmd=$cmd" --subset_id=$subset_id"
    cmd=$cmd" --report_freq=1000 --save_freq=2000"
    cmd=$cmd" --weight_decay=1e-8"
    cmd=$cmd" --save=$4"
    cmd=$cmd" --train_portion=$3"
    cmd=$cmd" --training_subset=$2/subset_$subset_id.txt"
    cmd=$cmd" --validation_file=/home/karima/cnn-data/val_files.txt"

    echo "running on gpu $gpu_id subset $subset_id"
    echo $cmd
    eval $cmd &
		pids[${subset_id}]=$!
		subset_id=$[$subset_id+1]
		if [ $subset_id -ge $n_subsets ]; then
			break
		fi
	done

	# wait for all pids
	for pid in ${pids[*]}; do
		wait $pid
	done

	unset pids
done
