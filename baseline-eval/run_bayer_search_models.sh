#!/bin/bash
GREEN_ASTS=$1
GEEN_WEIGHTS=$2
RETRAINED_DIR=$3
SEARCH_IDS=$4
SEARCH_DIR=$5

declare -a models

# Load file into array.
readarray models < ../bayer-baselines/$SEARCH_IDS

declare -a datasets=("kodak" "mcm" "moire" "hdrvdp")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
	cmd="python baseline-eval/compute-per-image-psnr.py \
		--rootdir=../bayer-baselines/$RETRAINED_DIR/ \
		--searchdir=$SEARCH_DIR \
		--model=$model \
		--dataset=$dataset \
		--compute \
		--crop=16 \
		--green_model_asts=$GREEN_ASTS \
		--green_model_weights=$GEEN_WEIGHTS \
		--bayer_search"
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
