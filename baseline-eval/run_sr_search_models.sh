#!/bin/bash
GREEN_ASTS=PARETO_GREEN_MODELS/PARETO_SUPERRES_GREEN_05-14/ast_files.txt
GEEN_WEIGHTS=PARETO_GREEN_MODELS/PARETO_SUPERRES_GREEN_05-14/weight_files.txt
RETRAINED_DIR=../sr-baselines/sr-joint-baselines/SEARCH_MODELS
SEARCH_IDS=search_model_ids.txt
SEARCH_DIR=/data/scratch/karima/sigasia-results/5-16-results/MULTINODE-SUPERRES-CHROMA-05-13-COMBINED/models

declare -a models

# Load file into array.
readarray models < ../sr-baselines/sr-joint-baselines/$SEARCH_IDS

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
		--green_model_asts=$GREEN_ASTS \
		--green_model_weights=$GEEN_WEIGHTS \
		--superres_search"
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