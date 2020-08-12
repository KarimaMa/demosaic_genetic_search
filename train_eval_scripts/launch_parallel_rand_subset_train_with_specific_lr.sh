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
epochs=${10}
lr_file=${11}
weight_decay=${12}

echo "using multires $use_multires"
echo "using multires2d $use_multires2d"
echo "using demosaicnet $use_demosaicnet"
echo "using ahd1d $use_ahd1d"
echo "using ahd2d $use_ahd2d"

start_gpu=1
end_gpu=3
subset_id=0

# read in the learning rates per subset
lrs=()
while IFS= read -r line; do
   lrs+=("$line")
done <$lr_file


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
    cmd=$cmd" --gpu=$gpu_id" 
    cmd=$cmd" --epochs=$epochs"
    cmd=$cmd" --subset_id=$subset_id"
    cmd=$cmd" --report_freq=100 --save_freq=2000"
    cmd=$cmd" --weight_decay=$weight_decay"
    cmd=$cmd" --save=$save_dir"
    cmd=$cmd" --train_portion=$train_portion"
    cmd=$cmd" --training_subset=$subset_dir/subset_$subset_id.txt"
    cmd=$cmd" --validation_file=/home/karima/cnn-data/val_files.txt"
    cmd=$cmd" --learning_rate=${lrs[$subset_id]}"

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
