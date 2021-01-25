#!/bin/bash
# Retrain one model on one GPU

ROOT=/mnt/ilcompf9d1/user/mgharbi/code/karima
CODE=$ROOT/demosaic_genetic_search

RETRAIN_DATA=$ROOT/retrain_data/DNET_CHROMA_MODEL_SEARCH_01-20
RETRAIN_LIST=$RETRAIN_DATA/pareto_model_ids.txt
RETRAIN_LOGS=$ROOT/retrain_logs/DNET_CHROMA_MODEL_SEARCH_01-20

let lineno=$1+1
MODEL_ID=$(sed -n $(printf $lineno)p $RETRAIN_LIST)

echo model $MODEL_ID at line $lineno of file $RETRAIN_LIST

CRASHED=$RETRAIN_LOGS/crashed
FINISHED=$RETRAIN_LOGS/finished
mkdir -p $CRASHED
mkdir -p $FINISHED

echo "Starting worker $1 for model $MODEL_ID"

echo "GPU info"
cat /proc/driver/nvidia/gpus/*/information

echo "Host NVIDIA driver"
cat /proc/driver/nvidia/version

echo "Testing GPU configuration..."
if ! nvidia-smi; then
    echo $(hostname) "job $1 nvidia-smi failed, aborting."
    echo $(date) >> $CRASHED/$MODEL_ID
    echo $(hostname) job $1  >> $CRASHED/$MODEL_ID
    echo "nvidia-smi failed" >>  $CRASHED/$MODEL_ID
    exit
fi
echo $(hostname) "nvidia-smi works"

echo "Testing PyTorch configuration..."
if ! python -c "import torch as th; print(th.zeros(3, 3).cuda())" ; then
    echo $(hostname) job $1  "PyTorch command failed, aborting"
    echo $(date) >> $CRASHED/$MODEL_ID
    echo $(hostname) job $1  >> $CRASHED/$MODEL_ID
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
        echo $(hostname)  job $1 "not enough space to copy data, aborting"
        echo $(date) >> $CRASHED/$MODEL_ID
        echo $(hostname)  job $1 >> $CRASHED/$MODEL_ID
        echo "no space to copy data" >>  $CRASHED/$MODEL_ID

        # TODO: try shm 
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

echo "Running $1, training model $MODEL_ID"
python $CODE_LOCAL/sys_run/retrain_one_model.py \
    --task_id=$1 \
    --gpu_id=0 \
    --model_info_dir=$RETRAIN_DATA/combined_models \
    --save=$ROOT/results/RETRAINED_DNET_CHROMA_MODEL_SEARCH_01-20 \
    --model_retrain_list=$RETRAIN_LIST \
    --epochs=5 \
    --learning_rate=0.003 \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --green_model_asts=$CODE_LOCAL/PARETO_GREEN_MODELS/dnetgreen_pareto_files/pareto_green_asts.txt \
    --green_model_weights=$CODE_LOCAL/PARETO_GREEN_MODELS/dnetgreen_pareto_files/pareto_green_weights.txt \
    --full_model

if [ $? -eq 0 ]
then
    echo $(hostname) "job completed successfully"
    echo $(date) >> $FINISHED/$MODEL_ID
    echo $(hostname) job $1 >> $FINISHED/$MODEL_ID
    echo "python completed successfully" >>  $FINISHED/$MODEL_ID
else
    echo $(hostname)  job $1 "crashed with an error"
    echo $(date) >> $CRASHED/$MODEL_ID
    echo $(hostname) job $1 >> $CRASHED/$MODEL_ID
    echo "python script return and error" >>  $CRASHED/$MODEL_ID
fi

# - task_id:  index 'into model_retrain_list'. 

# - model_info_dir: directory with the combined the model ast info, e.g. CHROMA_MODEL_SEARCH_01-20-INSERT/models/
# For example, every thing from CHROMA_MODEL_SEARCH_01-20-INSERT/models and CHROMA_MODEL_SEARCH_01-20-INSERT-NODE2/models
# gets copied into CHROMA_MODEL_SEARCH_01-20-INSERT-COMBINED/models, and we pass this directory

# - save: output directory e.g. RETRAINED_CHROMA_MODEL_SEARCH_01-20-INSERT/models/

# - training_file: full demosaicnet train.txt
# - validation_file: full demosaicnet val.txt

# --save=RETRAINED_GREEN_SEARCH_01-19-MULTIRES-DNET-SEEDS-INSERT-BINOP
# --model_retrain_list=pareto_model_ids.txt
# --model_info_dir=GREEN_SEARCH_01-19-MULTIRES-DNET-SEEDS-INSERT-BINOP-COMBINED/models
