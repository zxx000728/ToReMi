export TASK_NAME=cola

sudo CUDA_VISIBLE_DEVICES="0" python3 run_glue.py \
  --model_name_or_path /path/to/classification-two \
  --tokenizer_name openai-community/gpt2 \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 256 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir /path/to/output/cola/