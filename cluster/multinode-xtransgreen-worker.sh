#!/bin/zsh
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

WORKER=0
GPU=0

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
    --host=localhost
