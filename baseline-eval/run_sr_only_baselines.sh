#!/bin/bash
BASEDIR=../sr-baselines/sr-only-baselines

declare -a datasets=("BSD100" "Urban100" "Set14" "Set5" "hdrvdp" "moire")
# declare -a models=("drln.pt" "rcan.pt" "prosr.pt")

## now loop through the above array
for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
		cmd="python baseline-eval/gen-sr-joint-baseline-images.py \
			--datafile=../sr-baselines/${dataset}.txt \
			--output=$BASEDIR \
			--model=../sr-baselines/$model \
			--dataset=$dataset \
			--crop=12"

	 	echo $cmd >> "${model}_ran.txt"
	 	eval $cmd
	done
done

# run RAISR model
for dataset in "${datasets[@]}"
do
	cmd="python baseline-eval/gen-sr-joint-baseline-images.py \
		--datafile=../sr-baselines/${dataset}.txt \
		--output=$BASEDIR \
		--model=raisr \
		--dataset=$dataset \
		--crop=12"
 	
 	echo $cmd >> "${model}_ran.txt"
 	eval $cmd
done


# run FALSR models
# declare -a falsr_models=("FALSR-A.pb" "FALSR-B.pb" "FALSR-C.pb")

# for model in "${falsr_models[@]}"
# do
# 	for dataset in "${datasets[@]}"
# 	do
# 		cmd="python baseline-eval/run_falsr.py \
# 		--datafile=../sr-baselines/$dataset.txt \
# 		--output=$BASEDIR \
# 		--model=$BASEDIR/falsr_models/$model \
# 		--dataset=$dataset \
# 		--crop=12"
# 	 	echo $cmd >> "${model}_ran.txt"
# 	 	eval $cmd
# 	done
# done

