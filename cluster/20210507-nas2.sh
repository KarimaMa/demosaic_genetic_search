#!/bin/zsh

echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo "nvidia-smi failed, aborting"
    exit
fi

# This is needed to preempt all the GPUs on a machine...
if [ "$1" -ne "0" ]
then
    echo "Not the main job, looping"
    while true
    do
        printf "."
        sleep 10
    done
fi
# Only run actual job on job_id == 0
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

echo "Running job"
python $CODE_LOCAL/sys_run/run-using-queue.py \
    --experiment_name=nas \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --save=$ROOT/results/NAS-05-07-SEARCH-RUN1 \
    --cost_tiers="0,400 400,800 800,1600 1600,3200 3200,6400" \
    --pareto_sampling \
    --pareto_factor=3 \
    --machine=adobe \
    --mysql_auth=trisan4th \
    --mutations_per_generation=12 \
    --learning_rate=0.004 \
    --epochs=6 \
    --starting_model_id=4000 \
    --generations=40 \
    --tablename=adobegreen \
    --tier_size=20 \
    --nas_seed_model_files=seed_model_files/nas_seed_asts.txt \
    --nas_seed_model_psnrs=seed_model_files/nas_seed_psnrs.txt \
    --insertion_bias \
    --late_cdf_gen=9 \
    --max_footprint=30 \
    --resolution_change_factors=2,3 \
    --pixel_width=128 \
    --crop=16 \
    --nas
