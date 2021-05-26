#!/bin/bash

declare -a models=("MG_DNET_D15W64" "MG_DNET_D1W4" "MG_DNET_D2W8" "MG_DNET_D3W12" "MG_DNET_D5W16")

declare -a datasets=("kodak" "mcm" "moire" "hdrvdp")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
	cmd="python baseline-eval/compute-per-image-psnr.py \
		--rootdir=../bayer-baselines/DEMOSAICNET_BASELINES/ \
		--model=$model \
		--dataset=$dataset \
		--compute \
		--crop=16 \
		--demosaicnet"
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