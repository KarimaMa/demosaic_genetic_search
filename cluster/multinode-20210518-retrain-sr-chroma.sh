python $CODE_LOCAL/multinode_sys_run/retrain_one_model.py \
#!/bin/bash
# Retrain one model on one GPU
if [ "$#" -ne 3 ]; then
    echo "Script requries 3 inputs"
fi

JOB_NAME=sr-chroma-05-13

# Start_idx enables starting from other models than index 0
WORKER_ID=$1
START_IDX=$2
let TASK_ID=$WORKER_ID+$START_IDX

ROOT=/mnt/ilcompf9d1/user/mgharbi/code/karima
CODE=$ROOT/demosaic_genetic_search

RETRAIN_DATA=$ROOT/retrain_data/chroma-pareto-models/$JOB_NAME
RETRAIN_LIST=$RETRAIN_DATA/model_ids.txt
RETRAIN_LOGS=$ROOT/retrain_logs/$JOB_NAME

let lineno=$TASK_ID+1
MODEL_ID=$(sed -n $(printf $lineno)p $RETRAIN_LIST)

echo model $MODEL_ID at line $lineno of file $RETRAIN_LIST

CRASHED=$RETRAIN_LOGS/crashed
FINISHED=$RETRAIN_LOGS/finished
mkdir -p $CRASHED
mkdir -p $FINISHED

echo "Starting worker $WORKER_ID for model $MODEL_ID"

echo "GPU info"
cat /proc/driver/nvidia/gpus/*/information

echo "Host NVIDIA driver"
cat /proc/driver/nvidia/version

# Check if done already
if [[ -f $FINISHED/$MODEL_ID ]]
then
    echo "$FINISHED/$MODEL_ID exists, job is already done. Aborting."
    exit
fi

echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo $(hostname) "job $WORKER_ID nvidia-smi failed, aborting."
    echo $(date) >> $CRASHED/$MODEL_ID
    echo $(hostname) job $WORKER_ID  >> $CRASHED/$MODEL_ID
    echo "nvidia-smi failed" >>  $CRASHED/$MODEL_ID
    exit
fi
echo $(hostname) "nvidia-smi works"

echo "Testing PyTorch configuration..."
if ! python -c "import torch as th; print(th.zeros(3, 3).cuda())" ; then
    echo $(hostname) job $WORKER_ID  "PyTorch command failed, aborting"
    echo $(date) >> $CRASHED/$MODEL_ID
    echo $(hostname) job $WORKER_ID  >> $CRASHED/$MODEL_ID
    echo "PyTorch CUDA does not work" >>  $CRASHED/$MODEL_ID
    exit
fi
echo $(hostname) "PyTorch CUDA works"

echo "Memory"
free -h

echo "Disk space"
df -h | grep ssd

echo "SHM space"
df -h | grep shm

# TODO: check if job is done already

# DATA=$ROOT/data
DATA=/mnt/ilcompf8d1/data/demosaicnet
DATA_LOCAL=$DATA
# RuntimeError: Dataset filelist is invalid, coult not find /dev/shm/demosaic_genetic_search/train/moire/009/091892.png
mkdir -p /mnt/ssd/tmp/mgharbi
DATA_LOCAL=/mnt/ssd/tmp/mgharbi/demosaicnet
# DATA_LOCAL=/dev/shm/data
# rsync -av $DATA /dev/shm

if [ -d $DATA_LOCAL ]
then
    echo "Dataset already exist at $DATA_LOCAL"
    echo "rsyncing in case files are missing"
    rsync -av $DATA /mnt/ssd/tmp/mgharbi
    echo "done rsyncing data"
else
    freespace=$(df | grep ssd | awk '{print $4}')
    echo "Disk space" $freespace
    if (( $freespace < 90000000 )); then
        echo $(hostname)  job $WORKER_ID "not enough space to copy data, aborting"
        echo $(date) >> $CRASHED/$MODEL_ID
        echo $(hostname)  job $WORKER_ID >> $CRASHED/$MODEL_ID
        echo "no space to copy data" >>  $CRASHED/$MODEL_ID
        exit
    fi
    echo "Sufficient free space for data" $freespace
    echo "Copying dataset $DATA to $DATA_LOCAL"
    rsync -av $DATA /mnt/ssd/tmp/mgharbi
    echo "done copying data"
fi

echo "Copying code to local memory drive"
CODE_LOCAL=/dev/shm/demosaic_genetic_search
rsync -av $CODE /dev/shm

cd $CODE_LOCAL

echo "Installing python modules"
export PYTHONPATH=.:$PYTHONPATH
pip install -r $CODE_LOCAL/requirements.txt

echo "Running $WORKER_ID, training model $MODEL_ID"
    --task_id=$TASK_ID \
    --gpu_id=0 \
    --model_info_dir=$RETRAIN_DATA/models \
    --save=$ROOT/results/RETRAINED-SR-CHROMA-05-13 \
    --model_retrain_list=$RETRAIN_LIST \
    --epochs=5 \
    --learning_rate=0.004 \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --superres_rgb \
    --full_model \
    --green_model_asts=$CODE_LOCAL/PARETO_GREEN_MODELS/PARETO_SUPERRES_GREEN_05-14/ast_files.txt \
    --green_model_weights=$CODE_LOCAL/PARETO_GREEN_MODELS/PARETO_SUPERRES_GREEN_05-14/weight_files.txt \
    --train_timeout=25200

if [ $? -eq 0 ]
then
    echo $(hostname) "job completed successfully"
    echo $(date) >> $FINISHED/$MODEL_ID
    echo $(hostname) job $TASK_ID >> $FINISHED/$MODEL_ID
    echo "python completed successfully" >>  $FINISHED/$MODEL_ID
else
    echo $(hostname)  job $WORKER_ID "crashed with an error"
    echo $(date) >> $CRASHED/$MODEL_ID
    echo $(hostname) job $TASK_ID >> $CRASHED/$MODEL_ID
    echo "python script return and error" >>  $CRASHED/$MODEL_ID
fi
