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
python $CODE_LOCAL/sys_run/run.py \
    --experiment_name=green-multires-dnet-insert-binop \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --save=$ROOT/results/GREEN_SEARCH_01-19-MULTIRES-DNET-SEEDS-INSERT-BINOP-NODE2 \
    --cost_tiers="0,200 200,400 400,800 800,1600 1600,3200" \
    --pareto_sampling \
    --pareto_factor=3 \
    --machine=adobe \
    --mysql_auth=trisan4th \
    --mutations_per_generation=12 \
    --max_nodes=40 \
    --learning_rate=0.003 \
    --epochs=6 \
    --starting_model_id=2603 \
    --generations=20 \
    --tablename=adobegreen \
    --tier_size=20 \
    --green_seed_model_files=seed_model_files/green_seed_multires_dnet_asts.txt \
    --green_seed_model_psnrs=seed_model_files/green_seed_multires_dnet_psnrs.txt \
    --insertion_bias \
    --binop_change \
    --late_cdf_gen=9 \
    --restart_generation=11 \
    --restart_tier=4 \
    --tier_snapshot=$ROOT/results/GREEN_SEARCH_01-19-MULTIRES-DNET-SEEDS-INSERT-BINOP-NODE2/cost_tier_database/gen-11-snapshot-3 \
    --tier_db_snapshot=$ROOT/results/GREEN_SEARCH_01-19-MULTIRES-DNET-SEEDS-INSERT-BINOP-NODE2/cost_tier_database/TierDatabase-snapshot-20210119-074556 \
    --model_db_snapshot=$ROOT/results/GREEN_SEARCH_01-19-MULTIRES-DNET-SEEDS-INSERT-BINOP-NODE2/model_database/ModelDatabase-snapshot-20210119-074556 \
