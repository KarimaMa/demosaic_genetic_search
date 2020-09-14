#!/bin/bash
n_subsets=$1
subset_dir=$2
train_portion=$3
save_dir=$4
use_multires=$5
use_multires2d=$6
use_demosaicnet=$7
use_ahd1d=$8
use_ahd2d=$9
use_basic2d=${10}
epochs=${11}
learning_rate=${12}
weight_decay=${13}
model_initializations=${14}

echo "using multires $use_multires"
echo "using multires2d $use_multires2d"
echo "using demosaicnet $use_demosaicnet"
echo "using ahd1d $use_ahd1d"
echo "using ahd2d $use_ahd2d"
echo "using basic2d $use_basic2d"

start_gpu=0
end_gpu=3
subset_id=0

while [ $subset_id -lt $n_subsets ]; do
  for gpu_id in $(seq $start_gpu $end_gpu); do
    cmd="python3 train_eval_scripts/parallel_rand_subset_train.py"
    if [ $use_multires -eq 1 ]; then
      cmd=$cmd" --multires_model"
    fi
    if [ $use_multires2d -eq 1 ]; then
      cmd=$cmd" --multires_model2d"
    fi
    if [ $use_demosaicnet -eq 1 ]; then
      cmd=$cmd" --demosaicnet"
    fi
    if [ $use_ahd1d -eq 1 ]; then
      cmd=$cmd" --ahd1d"
    fi
    if [ $use_ahd2d -eq 1 ]; then
      cmd=$cmd" --ahd2d"
    fi
    if [ $use_basic2d -eq 1 ]; then
      cmd=$cmd" --basic_model2d"
    fi
    cmd=$cmd" --gpu=$gpu_id" 
    cmd=$cmd" --epochs=$epochs"
    cmd=$cmd" --subset_id=$subset_id"
    cmd=$cmd" --report_freq=50 --save_freq=2000"
    cmd=$cmd" --weight_decay=$weight_decay"
    cmd=$cmd" --save=$save_dir"
    cmd=$cmd" --train_portion=$train_portion"
    cmd=$cmd" --training_subset=$subset_dir/subset_$subset_id.txt"
    cmd=$cmd" --validation_file=/home/karima/cnn-data/val_files.txt"
    cmd=$cmd" --learning_rate=$learning_rate"
    cmd=$cmd" --validation_freq=50"
    cmd=$cmd" --model_initializations=$model_initializations"

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
