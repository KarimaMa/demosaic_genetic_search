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
    --experiment_name=green-gradhalide-dnet-insert-binop \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --save=$ROOT/results/GREEN_SEARCH_01-16-GRADHALIDE-DNET-SEEDS-INSERT-BINOP \
    --cost_tiers="0,200 200,400 400,800 800,1600 1600,3200" \
    --pareto_sampling \
    --pareto_factor=3 \
    --machine=adobe \
    --mysql_auth=trisan4th \
    --mutations_per_generation=12 \
    --max_nodes=40 \
    --learning_rate=0.003 \
    --epochs=6 \
    --starting_model_id=315 \
    --generations=20 \
    --tablename=adobegreen \
    --tier_size=20 \
    --green_seed_model_files=seed_model_files/green_seed_asts.txt \
    --green_seed_model_psnrs=seed_model_files/green_seed_psnrs.txt \
    --insertion_bias \
    --binop_change \
    --late_cdf_gen=9 \
    --restart_tier=2 \
    --restart_generation=6 \
    --tier_snapshot=$ROOT/results/GREEN_SEARCH_01-16-GRADHALIDE-DNET-SEEDS-INSERT-BINOP/cost_tier_database/gen-6-snapshot-1 \
    --tier_db_snapshot=$ROOT/results/GREEN_SEARCH_01-16-GRADHALIDE-DNET-SEEDS-INSERT-BINOP/cost_tier_database/TierDatabase-snapshot-20210118-025142 \
    --model_db_snapshot=$ROOT/results/GREEN_SEARCH_01-16-GRADHALIDE-DNET-SEEDS-INSERT-BINOP/model_database/ModelDatabase-snapshot-20210118-025142
