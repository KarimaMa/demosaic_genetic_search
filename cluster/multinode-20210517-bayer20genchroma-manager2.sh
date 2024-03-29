#!/bin/zsh

echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo "nvidia-smi failed, aborting"
    exit
fi

echo "Starting worker $1"

ROOT=/mnt/ilcompf9d1/user/mgharbi/code/karima

echo "Copying code to local memory drive"
CODE=$ROOT/demosaic_genetic_search
CODE_LOCAL=/dev/shm/demosaic_genetic_search
rsync -av $CODE /dev/shm

cd $CODE_LOCAL

echo "Installing python modules"
export PYTHONPATH=.:$PYTHONPATH
pip install -r $CODE_LOCAL/requirements.txt

echo "Copying data to local memory drive"
DATA=$ROOT/data
DATA_LOCAL=/dev/shm/data
rsync -av $DATA /dev/shm

STARTID=4000
RUN=1
PORT=2001
DATE=17

echo "Running job"
python $CODE_LOCAL/multinode_sys_run/manager.py \
    --experiment_name=bayer-chroma \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --save=$ROOT/results/MULTINODE-BAYER-20GEN-CHROMA-05-$DATE-RUN$RUN \
    --cost_tiers="0,300 300,600 600,1200 1200,2400 2400,4800" \
    --pareto_sampling \
    --pareto_factor=3 \
    --machine=adobe \
    --mysql_auth=trisan4th \
    --mutations_per_generation=12 \
    --learning_rate=0.004 \
    --epochs=6 \
    --starting_model_id=$STARTID \
    --generations=40 \
    --tablename=adobegreen \
    --tier_size=20 \
    --insertion_bias \
    --late_cdf_gen=9 \
    --max_footprint=30 \
    --resolution_change_factors=2,3 \
    --pixel_width=128 \
    --crop=16 \
    --max_nodes=50 \
    --max_subtree_size=25 \
    --train_timeout=1200 \
    --full_model \
    --port=$PORT \
    --green_model_asts=$CODE_LOCAL/PARETO_GREEN_MODELS/PARETO_BAYER_GREEN_20GEN_05-14/ast_files.txt \
    --green_model_weights=$CODE_LOCAL/PARETO_GREEN_MODELS/PARETO_BAYER_GREEN_20GEN_05-14/weight_files.txt \
    --chroma_seed_model_files=$CODE_LOCAL/seed_model_files/bchroma-flagship-seed-asts.txt \
    --chroma_seed_model_psnrs=$CODE_LOCAL/seed_model_files/bchroma-flagship-seed-psnrs.txt 
