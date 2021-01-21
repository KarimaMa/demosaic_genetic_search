#!/bin/zsh


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

echo "Installing python modules"
export PYTHONPATH=.:$PYTHONPATH
pip install -r $CODE_LOCAL/requirements.txt

echo "Copying data to local memory drive"
DATA=$ROOT/data
DATA_LOCAL=/dev/shm/data
rsync -av $DATA /dev/shm

echo "Running job"


python $CODE_LOCAL/sys_run/run.py \
    --experiment_name=chroma-search \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --save=$ROOT/results/CHROMA_MODEL_SEARCH_01-20-INSERT-NODE2 \
    --cost_tiers="0,250 250,500 500,1000 1000,2000 2000,4000" \
    --pareto_sampling \
    --pareto_factor=3 \
    --machine=adobe \
    --mysql_auth=trisan4th \
    --mutations_per_generation=12 \
    --max_nodes=40 \
    --learning_rate=0.003 \
    --epochs=6 \
    --starting_model_id=2000 \
    --generations=20 \
    --tablename=chroma \
    --tier_size=20 \
    --full_model \
    --insertion_bias \
    --green_model_asts=$CODE_LOCAL/PARETO_GREEN_MODELS/green-01-18-pareto-files/pareto_green_asts.txt \
    --green_model_weights=$CODE_LOCAL/PARETO_GREEN_MODELS/green-01-18-pareto-files/pareto_green_weights.txt \
    --chroma_seed_model_files=$CODE_LOCAL/seed_model_files/chroma-01-20-i-green-01-18-im-asts.txt \
    --chroma_seed_model_psnrs=$CODE_LOCAL/seed_model_files/chroma-01-20-i-green-01-18-im-psnrs.txt 