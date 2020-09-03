n_subsets=$1
subset_dir=$2
kfold_id=$3
kfold_size=$4
num_images=$5
save_dir=$6
use_multires=$7
use_multires2d=$8
use_demosaicnet=$9
use_ahd1d=${10}
use_ahd2d=${11}
use_basic2d=${12}
epochs=${13}
learning_rate=${14}
weight_decay=${15}
model_initializations=${16}

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
    cmd="python3 train_eval_scripts/kfold_study.py"
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
    cmd=$cmd" --report_freq=10 --save_freq=2000"
    cmd=$cmd" --weight_decay=$weight_decay"
    cmd=$cmd" --save=$save_dir"
    cmd=$cmd" --kfold_size=$kfold_size"
    cmd=$cmd" --kfold_id=$kfold_id"
    cmd=$cmd" --num_images=$num_images"
    cmd=$cmd" --training_subset=$subset_dir/subset_$subset_id.txt"
    cmd=$cmd" --learning_rate=$learning_rate"
    cmd=$cmd" --validation_freq=100"
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
