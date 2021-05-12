#!/bin/zsh

echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo "nvidia-smi failed, aborting"
    exit
fi

WORKER=$1  # Condor passes the woker ID as argument
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

GPU=0
PORT=2001

echo "Running job"
python $CODE_LOCAL/multinode_sys_run/worker.py \
    --experiment_name=xtrans-green \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --save=$ROOT/results/TEST-MULTINODE-XGREEN \
    --learning_rate=0.004 \
    --epochs=6 \
    --green_seed_model_files=xtrans-seed-model-files/green_seed_asts.txt \
    --green_seed_model_psnrs=xtrans-seed-model-files/green_seed_psnrs.txt \
    --crop=12 \
    --xtrans_green \
    --gpu_id=$GPU \
    --worker_id=$WORKER \
    --port=$PORT \
    --host=ilcomp6u
