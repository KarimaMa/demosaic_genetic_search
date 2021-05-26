#!/bin/bash

declare -a models

# Load file into array.
readarray models < ../bayer-baselines/nas_model_ids.txt

echo $models

declare -a datasets=("kodak" "mcm" "moire" "hdrvdp")
#declare -a datasets=("mcm")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
	cmd="python baseline-eval/compute-per-image-psnr.py \
		--rootdir=../bayer-baselines/NAS_MODELS/ \
    --searchdir=/data/scratch/karima/sigasia-results/5-16-results/MULTINODE-NAS-05-13-COMBINED/models \
		--model=$model \
		--dataset=$dataset \
		--compute \
		--crop=16 \
		--nas_search"
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
