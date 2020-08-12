#!/bin/bash
n_subsets=$1
subset_dir=$2
train_portion=$3
save_dir=$4
use_multires=$5
use_multires2d=$6
use_demosaicnet=$7
use_ahd=$8
use_ahd2d=$9
max_learning_rate=${10}
weight_decay=${11}
divergence_threshold=${12}
min_lr=${13}
eps=${14}
decision_point=${15}

echo "using multires $use_multires"
echo "using multires2d $use_multires2d"
echo "using demosaicnet $use_demosaicnet"
echo "using ahd $use_ahd"
echo "using ahd2d $use_ahd2d"

start_gpu=1
end_gpu=3
subset_id=0

while [ $subset_id -lt $n_subsets ]; do
  for gpu_id in $(seq $start_gpu $end_gpu); do
    cmd="python3 data_distillation/find_model_lr_on_subset.py"
    if [ $use_multires -eq 1 ]; then
      cmd=$cmd" --multires_model"
    fi
    if [ $use_multires2d -eq 1 ]; then
      cmd=$cmd" --multires_model2d"
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
