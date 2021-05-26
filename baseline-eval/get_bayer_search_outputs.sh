#!/bin/bash
GREEN_ASTS=$1
GREEN_WEIGHTS=$2
ROOTDIR=$3
SEARCH_DIR=$4
OUTDIR=$5

declare -a models

# Load file into array.
readarray models < ../bayer-baselines/$ROOTDIR/model_ids.txt

declare -a datasets=("kodak" "mcm" "moire" "hdrvdp")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
	cmd="python baseline-eval/get-per-image-outputs.py \
		--rootdir=../bayer-baselines/$ROOTDIR/ \
		--searchdir=$SEARCH_DIR \
		--model=$model \
		--dataset=$dataset \
		--compute \
		--crop=16 \
		--green_model_asts=$GREEN_ASTS \
		--green_model_weights=$GREEN_WEIGHTS \
		--search_models"
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