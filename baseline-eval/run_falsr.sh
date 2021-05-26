#!/bin/bash

BASEDIR=../sr-baselines/sr-only-baselines

declare -a datasets=("BSD100" "Urban100" "Set5" "Set14" "hdrvdp" "moire")

declare -a models=("FALSR-A.pb" "FALSR-B.pb" "FALSR-C.pb")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
		python baseline-eval/run_falsr.py \
		--datafile=../sr-baselines/$dataset.txt \
		--output=$BASEDIR \
		--model=$BASEDIR/falsr_models/$model \
		--dataset=$dataset \
		--crop=12
		echo $cmd
		eval $cmd
	done
done