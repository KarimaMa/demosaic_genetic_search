#!/bin/zsh

echo $(hostname)
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

REMOTE_HOST=$2
PORT=$3

echo "Running worker, remote host is:" $REMOTE_HOST "port:" $PORT
python $CODE_LOCAL/multinode_sys_run/worker.py \
    --gpu_id=$GPU \
    --worker_id=$WORKER \
    --port=$PORT \
    --host=$REMOTE_HOST
