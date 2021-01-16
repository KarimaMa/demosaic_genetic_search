#!/bin/zsh

echo "Starting worker $1"

ROOT=/mnt/ilcompf9d1/user/mgharbi/code/karima

cd $ROOT/demosaic_genetic_search

echo "Installing python modules"
export PYTHONPATH=.:$PYTHONPATH
pip install -r requirements.txt

echo "Copying data to local memory drive"
DATA=/mnt/ilcompf9d1/user/mgharbi/code/karima/data
DATA_LOCAL=/dev/shm/data
rsync -av $DATA /dev/shm


echo "Running job"
python sys_run/run.py \
    --experiment_name=rgb8chan \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --save=$ROOT/results/RGB8CHAN_MODEL_SEARCH_01-15-NODE2 \
    --cost_tiers="0,250 250,500 500,1000 1000,2000 2000,4000" \
    --pareto_sampling \
    --pareto_factor=3 \
    --machine=adobe \
    --mysql_auth=trisan4th \
    --mutations_per_generation=12 \
    --max_nodes=40 \
    --learning_rate=0.003 \
    --epochs=6 \
    --starting_model_id=4000 \
    --generations=40 \
    --tablename=rgb8chan \
    --tier_size=20 \
    --rgb8chan_seed_model_files=seed_model_files/rgb8chan_seed_asts.txt \
    --rgb8chan_seed_model_psnrs=seed_model_files/rgb8chan_seed_psnrs.txt \
    --rgb8chan \
    --insertion_bias \
    --binop_change \
    --late_cdf_gen=9
