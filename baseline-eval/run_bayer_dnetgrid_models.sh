#!/bin/bash

declare -a models

# Load file into array.
readarray models < ../bayer-baselines/dnet_grid_model_ids.txt

echo $models

declare -a datasets=("kodak" "mcm" "moire" "hdrvdp")
#declare -a datasets=("mcm")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
	cmd="python baseline-eval/compute-per-image-psnr.py \
		--rootdir=../bayer-baselines/DNET_GRIDSEARCH_MODELS/ \
		--model=$model \
		--dataset=$dataset \
		--compute \
		--crop=16 \
		--dnetgrid_models"
	if [ $dataset == "hdrvdp" ] || [ $dataset == "moire" ]
	then
		cmd=$cmd" --eval_imagedir=../cnn-data/test/$dataset/000/"
	else
		cmd=$cmd" --eval_imagedir=../bayer-baselines/datasets/$dataset"
	fi
	echo $cmd
	eval $cmd
	done
done