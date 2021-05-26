#!/bin/bash
OUTPUTDIR=../sr-baselines/sr-joint-baselines

declare -a datasets=("Set5" "BSD100" "Urban100" "Set14" "hdrvdp" "moire")
declare -a models=("drln.pt" "rcan.pt" "prosr.pt" "tenet.pt")

for model in "${models[@]}"
do
	for dataset in "${datasets[@]}"
	do
		cmd="python baseline-eval/gen-sr-joint-baseline-images.py \
			--datafile=../sr-baselines/${dataset}.txt \
			--output=$OUTPUTDIR \
			--model=../sr-baselines/$model \
			--dataset=$dataset \
			--crop=12 \
			--joint"
	 	echo $cmd >> "sr_joint_${model}_ran.txt"
	 	eval $cmd
	done
done

# run RAISR model
# echo "RUNNING RAISR JOINT"
# for dataset in "${datasets[@]}"
# do
# 	cmd="python baseline-eval/gen-sr-joint-baseline-images.py \
# 		--datafile=../sr-baselines/${dataset}.txt \
# 		--output=$BASEDIR \
# 		--model=raisr \
# 		--dataset=$dataset \
# 		--crop=12 \
# 		--joint"
 	
#  	echo $cmd >> "${model}_ran.txt"
#  	eval $cmd
# done


# run FALSR models
# echo "RUNNING FALSR JOINT"
# declare -a falsr_models=("FALSR-A.pb" "FALSR-B.pb" "FALSR-C.pb")

# for model in "${falsr_models[@]}"
# do
# 	for dataset in "${datasets[@]}"
# 	do
# 		cmd="python baseline-eval/run_falsr.py \
# 		--datafile=../sr-baselines/${dataset}.txt \
# 		--output=$OUTPUTDIR \
# 		--model=../sr-baselines/falsr_models/$model \
# 		--dataset=$dataset \
# 		--crop=12 \
# 		--joint"
# 	 	echo $cmd >> "${model}_ran.txt"
# 	 	eval $cmd
# 	done
# done

# # run espcn models
# echo "RUNNING ESPCN JOINT"
# for model in "${falsr_models[@]}"
# do
# 	for dataset in "${datasets[@]}"
# 	do
# 		cmd="python baseline-eval/run_espcn.py \
# 		--datafile=../sr-baselines/${dataset}.txt \
# 		--output=$OUTPUTDIR \
# 		--model=../sr-baselines/espcn_2x.pth \
# 		--dataset=$dataset \
# 		--crop=12 \
# 		--joint"
# 	 	echo $cmd >> "${model}_ran.txt"
# 	 	eval $cmd
# 	done
# done