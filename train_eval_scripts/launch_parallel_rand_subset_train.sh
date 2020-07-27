#!/bin/bash
n_procs=10
n_gpus=3
proc_id=0

while [ $proc_id -lt $n_procs ]; do
  for gpu_id in $(seq 0 $n_gpus); do
		python3 parallel_rand_subset_train.py --gpu=$gpu_id --subset_id=$proc_id \
		--training_subset="/home/karima/cnn-data/shuf"$proc_id"train_files.txt" \
		--validation_file=/home/karima/cnn-data/val_files.txt &
    echo $gpu_id $proc_id
		pids[${proc_id}]=$!
		proc_id=$[$proc_id+1]
		if [ $proc_id -ge $n_procs ]; then
			break
		fi
	done

	# wait for all pids
	for pid in ${pids[*]}; do
		wait $pid
	done

	unset pids
done
