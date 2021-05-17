#!/bin/bash

SEARCH_MODELS=$1
INFODIR=$2

python pareto-model-selection/select_green_models.py \
	--search_models=$SEARCH_MODELS \
	--num_buckets=10 \
	--infodir=$INFODIR 