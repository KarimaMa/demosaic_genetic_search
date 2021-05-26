#!/bin/bash

declare -a models=("MODEL-K5F2" "MODEL-K5F9" "MODEL-K7F15")

declare -a datasets=("kodak" "mcm" "moire" "hdrvdp")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
	cmd="python baseline-eval/compute-per-image-psnr.py \
		--rootdir=../bayer-baselines/GRADHALIDE_BASELINES/ \
		--model=$model \
		--dataset=$dataset \
		--compute \
		--crop=16 \
		--gradienthalide"
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