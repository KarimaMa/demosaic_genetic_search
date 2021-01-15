#!/bin/zsh

echo "Starting worker $1"

ROOT=/mnt/ilcompf9d1/user/mgharbi/code/karima

cd $ROOT/demosaic_genetic_search

echo "Installing python modules"
export PYTHONPATH=.:$PYTHONPATH
pip install -r requirements.txt

echo "Copying data to local memory drive"
DATA=/mnt/ilcompf9d1/user/mgharbi/code/karima/data
DATA_LOCAL=/dev/shm/data/sample_files.txt
rsync -av $DATA /dev/shm

echo "Running job"
python sys_run/run.py \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --save=$ROOT/GREEN_MODEL_SEARCH_01-15-MULTIRES-DNET-SEEDS \
    --cost_tiers="0,200 200,400 400,800 800,1600 1600,3200" \
    --pareto_sampling \
    --pareto_factor=3 \
    --machine=adobe \
    --mysql_auth=trisan4th \
    --mutations_per_generation=12 \
    --max_nodes=40 \
    --learning_rate=0.003 \
    --epochs=6 \
    --starting_model_id=0 \
    --generations=20 \
    --tablename=adobegreen \
    --tier_size=20 \
    --green_seed_model_files=seed_model_files/green_seed_multires_dnet_asts.txt \
    --green_seed_model_psnrs=seed_model_files/green_seed_multires_dnet_psnrs.txt \
    --insertion_bias \
    --late_cdf_gen=9
