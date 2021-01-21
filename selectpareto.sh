#!/bin/bash

DATADIR=$1
green_models=$2
RETRAINDIR=$3
SEED_FILES_DIR=$4
SELECTED_MODELS_DIR=$5


base_cmd="python3 select_pareto_models.py \
			--retrained_pareto_dir=${RETRAINDIR} \
			--model_inits=3 \
			--pareto_ids=${DATADIR}/pareto_model_ids.txt \
			--seed_ast_file=pareto_green_asts.txt \
			--seed_psnr_file=pareto_green_psnrs.txt \
			--seed_weight_file=pareto_green_weights.txt \
			--seed_files_dir=${SEED_FILES_DIR} \
			--selected_models_dir=${SELECTED_MODELS_DIR}"

if [ $green_models ]; then
	cost_tiers="50,200 200,400 400,800 800,1600 1600,3200"
else
	cost_tiers="50,250 250,500 500,1000 1000,2000 2000,4000"
fi

base_cmd=${base_cmd}" --cost_tiers='${cost_tiers}'"

echo $base_cmd
eval $base_cmd	
