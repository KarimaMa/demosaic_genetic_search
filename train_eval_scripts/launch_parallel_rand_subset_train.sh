#!/bin/bash
n_procs=10
n_gpus=3
proc_id=0

while [ $proc_id -lt $n_procs ]; do
  for gpu_id in $(seq 0 $n_gpus); do
		python3 train_eval_scripts/parallel_rand_subset_train.py --gpu=$gpu_id --subset_id=$proc_id \
    --report_freq=1000 \
    --save_freq=2000 \
    --weight_decay=1e-8 \
    --save=BASIC_GREEN_SUBSET_TRAIN_SMALL_WEIGHT_DECAY \
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
