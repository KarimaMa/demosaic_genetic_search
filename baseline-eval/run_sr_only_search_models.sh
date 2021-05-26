#!/bin/bash
RETRAINED_DIR=../sr-baselines/sr-only-baselines/SEARCH_MODELS
SEARCH_IDS=search_model_ids.txt
SEARCH_DIR=/data/scratch/karima/sigasia-results/5-16-results/MULTINODE-SR-ONLY-05-17-COMBINED/models

declare -a models

# Load file into array.
readarray models < ../sr-baselines/sr-only-baselines/$SEARCH_IDS

declare -a datasets=("BSD100" "Urban100" "Set5" "Set14" "hdrvdp" "moire")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
	cmd="python baseline-eval/compute-per-image-psnr.py \
		--rootdir=$RETRAINED_DIR/ \
		--searchdir=$SEARCH_DIR \
		--model=$model \
		--dataset=$dataset \
		--compute \
		--crop=12 \
		--superres_only_search"
	if [ $dataset == "hdrvdp" ] || [ $dataset == "moire" ]
	then
		cmd=$cmd" --eval_imagedir=../cnn-data/test/$dataset/000/"
	else
		cmd=$cmd" --eval_imagedir=../sr-baselines/SelfExSR/data/$dataset/image_SRF_2"
	fi
	echo $cmd
	eval $cmd
	done
done