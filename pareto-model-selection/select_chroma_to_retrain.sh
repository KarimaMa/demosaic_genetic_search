#!/bin/bash
MODEL_DIR=$1
OUTDIR=$2

python pareto-model-selection/collect_pareto_models.py \
	--search_models=$MODEL_DIR \
	--outdir=$OUTDIR \
 	--n=100
