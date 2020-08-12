#!/bin/bash
n_subsets=$1
subset_dir=$2
train_portion=$3
save_dir=$4
use_multires=$5
use_demosaicnet=$6
use_ahd=$7
use_ahd2d=$8
max_learning_rate=$9
weight_decay=${10}
divergence_threshold=${11}
min_lr=${12}
eps=${13}
decision_point=${14}

echo "using multires $use_multires"
echo "using demosaicnet $use_demosaicnet"
echo "using ahd $use_ahd"
echo "using ahd2d $use_ahd2d"

start_gpu=0
end_gpu=0
subset_id=0

while [ $subset_id -lt $n_subsets ]; do
  for gpu_id in $(seq $start_gpu $end_gpu); do
    cmd="python3 data_distillation/find_model_lr_on_subset.py"
    if [ $use_multires -eq 1 ]; then
      cmd=$cmd" --multires_model"
    fi
    if [ $use_demosaicnet -eq 1 ]; then
      cmd=$cmd" --demosaicnet"
    fi
    if [ $use_ahd -eq 1 ]; then
      cmd=$cmd" --ahd"
    fi
    if [ $use_ahd2d -eq 1 ]; then
      cmd=$cmd" --ahd2d"
    fi
    cmd=$cmd" --gpu=$gpu_id" 
    cmd=$cmd" --subset_id=$subset_id"
    cmd=$cmd" --save_freq=100"
    cmd=$cmd" --weight_decay=$weight_decay"
    cmd=$cmd" --save=$save_dir"
    cmd=$cmd" --train_portion=$train_portion"
    cmd=$cmd" --training_file=$subset_dir/subset_$subset_id.txt"
    cmd=$cmd" --validation_file=/home/karima/cnn-data/val_files.txt"
    cmd=$cmd" --max_learning_rate=$max_learning_rate"
    cmd=$cmd" --divergence_threshold=$divergence_threshold"
    cmd=$cmd" --min_lr=$min_lr"
    cmd=$cmd" --eps=$eps"
    cmd=$cmd" --decision_point=$decision_point"

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
