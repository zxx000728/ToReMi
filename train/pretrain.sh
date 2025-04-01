#!/bin/bash
CURRENT_DIR=$(cd $(dirname $0); pwd)
model="gpt"
DATASET_DIR=/path/to/dataset_dir
TRAIN_DATASET_DIR=/path/to/dataset_dir/train_data
MAXLEN=1024
MODEL_NAME_OR_PATH=gpt2
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=32
GRAD_ACCUM=4
WD=0.01
LR=1e-4
WARMUP=500

OUTPUT_DIR=/path/to/output
TASK_NAME=task_name

sudo accelerate launch --config_file $CURRENT_DIR/accelerate_config.yaml $CURRENT_DIR/run_pipeline.py \
        --pipeline_step pretrain \
        --dataset_dir ${DATASET_DIR} \
        --train_dataset_dir ${TRAIN_DATASET_DIR} \
        --from_scratch \
        --max_train_steps 8000 \
        --steps_to_eval 100 \
        --steps_to_save 1000 \
        --steps_to_log 100 \
        --steps_to_adjust_weight 100 \
        --max_length ${MAXLEN} \
        --pad_to_max_length \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --output_dir $OUTPUT_DIR \
        --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
        --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
        --gradient_accumulation_steps ${GRAD_ACCUM} \
        --cuda_devices 0,1,2,3,4,5 \
        --task_name ${TASK_NAME} \
        --save_final \
        --weight_decay $WD \
        --learning_rate $LR \
        --num_warmup_steps $WARMUP \
        --seed 18 \
        --model ${model} \
        --with_labels
