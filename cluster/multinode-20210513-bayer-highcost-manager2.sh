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
DATE=13

echo "Running job"
python $CODE_LOCAL/multinode_sys_run/manager.py \
    --experiment_name=nas \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --save=$ROOT/results/MULTINODE-BAYER-GREEN-HIGH-COST-05-$DATE-RUN$RUN \
    --cost_tiers="0,1000 1000,2000 2000,4000 4000,8000 8000,16000" \
    --pareto_sampling \
    --pareto_factor=3 \
    --machine=adobe \
    --mysql_auth=trisan4th \
    --mutations_per_generation=12 \
    --learning_rate=0.004 \
    --epochs=6 \
    --starting_model_id=$STARTID \
    --generations=20 \
    --tablename=adobegreen \
    --tier_size=20 \
    --green_seed_model_files=seed_model_files/green_seed_asts-05-06-2021.txt \
    --green_seed_model_psnrs=seed_model_files/green_seed_psnrs-05-06-2021.txt \
    --insertion_bias \
    --late_cdf_gen=9 \
    --max_footprint=64 \
    --resolution_change_factors=2,3 \
    --pixel_width=128 \
    --crop=0 \
    --max_nodes=100 \
    --max_subtree_size=50 \
    --train_timeout=3000 \
    --port=$PORT
