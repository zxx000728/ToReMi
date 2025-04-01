#!/bin/bash
DATASET_DIR=/path/to/dataset_dir
TRAIN_DATASET_DIR=/path/to/dataset_dir/train_data
MAXLEN=1024
OUTPUT_DIR=/path/to/output
MODEL_NAME_OR_PATH=gpt2
TASK_NAME=preprocess-datasets

sudo accelerate launch --config_file ./accelerate_preprocess_config.yaml run_pipeline.py \
    --pipeline_step preprocess \
    --from_scratch \
    --dataset_dir ${DATASET_DIR} \
    --train_dataset_dir ${TRAIN_DATASET_DIR} \
    --task_name ${TASK_NAME} \
    --max_length ${MAXLEN} \
    --pad_to_max_length \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --cuda_devices 0,1,2,3,4,5,6,7 \
    --output_dir $OUTPUT_DIR \
    --seed 18 \
    --with_labels
