#!/bin/bash
# Retrain one model on one GPU
if [ "$#" -ne 2 ]; then
    echo "Script requries 2 input"
    exit
fi

# Start_idx enables starting from other models than index 0
WORKER_ID=$1
GPU=$(($2-1))
START_IDX=0
let TASK_ID=$WORKER_ID+$START_IDX

JOB_NAME=xtrans-chroma-05-15

ROOT=/home/code-base/scratch_space/karima
CODE=/home/mgharbi/demosaic_genetic_search

RETRAIN_DATA=$ROOT/retrain_data/chroma-pareto-models/$JOB_NAME
RETRAIN_LIST=$RETRAIN_DATA/model_ids.txt
RETRAIN_LOGS=$ROOT/retrain_logs/$JOB_NAME

let lineno=$TASK_ID+1
MODEL_ID=$(sed -n $(printf $lineno)p $RETRAIN_LIST)

echo Task $TASK_ID, GPU $GPU, worker $WORKER_ID: model $MODEL_ID at line $lineno of $RETRAIN_LIST

CRASHED=$RETRAIN_LOGS/crashed
FINISHED=$RETRAIN_LOGS/finished
mkdir -p $CRASHED
mkdir -p $FINISHED

# Check if done already
if [[ -f $FINISHED/$MODEL_ID ]]
then
    echo "$FINISHED/$MODEL_ID exists, job is already done. Aborting."
    exit
fi

DATA_LOCAL=/dev/shm/demosaicnet

python $CODE/multinode_sys_run/retrain_one_model.py \
    --task_id=$TASK_ID \
    --gpu_id=$GPU \
    --model_info_dir=$RETRAIN_DATA/models \
    --save=$ROOT/results/RETRAINED-XTRANS-CHROMA-05-15 \
    --model_retrain_list=$RETRAIN_LIST \
    --epochs=5 \
    --learning_rate=0.004 \
    --training_file=$DATA_LOCAL/train.txt \
    --validation_file=$DATA_LOCAL/val.txt \
    --xtrans_chroma \
    --full_model \
    --green_model_asts=$CODE/PARETO_GREEN_MODELS/PARETO_XTRANS_GREEN_AZURE-05-10/ast_files.txt \
    --green_model_weights=$CODE/PARETO_GREEN_MODELS/PARETO_XTRANS_GREEN_AZURE-05-10/weight_files.txt \
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
