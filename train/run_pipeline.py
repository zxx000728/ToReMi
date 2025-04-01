import argparse
import traceback
import logging
import math
import json
import os
import datasets
from datasets import load_dataset, load_from_disk, concatenate_datasets
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
import wandb
import transformers
from datetime import datetime
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    DataCollatorForLanguageModeling,
    BartForConditionalGeneration,
    get_scheduler,
    set_seed,
)
from transformers.models.gpt2 import GPT2LMHeadModel
from collator import DataCollatorForDenoisingTasks
from model import BertForMaskedLM
from train.trainer import PretrainTrainer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_file", type=str, default=None,
                        help="The name of the tokenizer file or path")
    parser.add_argument("--pipeline_step",
                        type=str, default='preprocess', help="step of the pipeline - preprocess, pretrain")
    parser.add_argument("--save_final", action="store_true",
                        help="save the final checkpoint")
    parser.add_argument("--from_scratch", action="store_true",
                        help="training from scratch")
    parser.add_argument("--from_ckpt",
                        type=str, default=None, help="restore the model training process from a checkpoint")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="The name of the directory storing the datasets")
    parser.add_argument("--train_dataset_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str,
                        default='cache', help="path to cache directory")
    parser.add_argument("--preprocessing_num_workers",
                        type=int, default=None, help="Number of preprocessors")
    parser.add_argument("--steps_to_log", type=int,
                        default=None, help="Num steps to log training info")
    parser.add_argument("--steps_to_eval", type=int, default=None,
                        help="Num steps to evaluate on the dev set")
    parser.add_argument("--steps_to_save", type=int,
                        default=None, help="Num steps to save the checkpoint")
    parser.add_argument("--steps_to_adjust_weight", type=int,
                        default=None, help="Num steps to adjust label weights")
    parser.add_argument("--max_grad_norm", type=float,
                        default=1.0, help="Max gradient norm")
    parser.add_argument("--task_name", type=str, default=None,
                        help="The name of the task to train on.")
    parser.add_argument("--max_length", type=int, default=1024,
                        help=(
                            "The maximum total input sequence length after tokenization. "
                            "Sequences longer than this will be truncated, "
                            "sequences shorter will be padded if `--pad_to_max_length` is passed."
                        ))
    parser.add_argument("--pad_to_max_length", action="store_true",
                        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--model_name_or_path", type=str,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--config_dir", type=str, default=None,
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8,
                        help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8,
                        help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=None,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=0,
                        help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear",
                        help="The scheduler type to use."
                        # choices=["linear", "cosine", "cosine_with_restarts",
                        # "polynomial", "constant", "constant_with_warmup"]
                        )
    parser.add_argument("--num_warmup_steps", type=float, default=10000,
                        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--cuda_devices", type=str,
                        default='0', help="visible cuda devices.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=18,
                        help="A seed for reproducible training.")
    parser.add_argument("--model", type=str)
    parser.add_argument("--with_labels", action="store_true")
    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None:
        raise ValueError("Need a task name.")

    if args.model_name_or_path is None:
        assert args.from_scratch, "No model name or path is provided but trying to initialize from a pre-trained weight"

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args"), "w") as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")

    return args


def get_logger(args, accelerator=None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator is not None:
        logger.info(accelerator.state)

    if args.output_dir is not None:
        now = datetime.now()
        timestamp = str(datetime.timestamp(now))
        log_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        logfile = os.path.join(log_dir, f"log_{timestamp}")

        if accelerator is not None and accelerator.is_main_process:
            os.mknod(logfile)
            fh = logging.FileHandler(logfile, mode='w')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    if accelerator is None:
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    else:
        logger.setLevel(
            logging.INFO if accelerator.is_main_process else logging.ERROR)
        if accelerator.is_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    return logger


def get_dataset(args):
    dataset_dir = args.dataset_dir
    train_file = args.train_dataset_dir
    train_dataset = load_from_disk(train_file)
    data_files = {
        "dev": f"{dataset_dir}/dev.json",
        "test": f"{dataset_dir}/test.json",
    }
    raw_datasets = load_dataset(
        "json", data_files=data_files, cache_dir=f"{dataset_dir}/{args.cache_dir}")
    raw_datasets["train"] = train_dataset
    return raw_datasets


def preprocess(args, raw_datasets, tokenizer, logger, preprocessed_cache):
    logger.info("Start preprocessing datasets")
    padding = "max_length" if args.pad_to_max_length else False
    metadatas = set()

    def tokenize_function(examples):
        # Remove empty lines & tokenize the texts
        if "labels" in examples.keys():
            filtered_data = [
                (content, metadata, cluster, labels)
                for content, metadata, cluster, labels in zip(
                    examples["text"], examples["embeddings"], examples["cluster"], examples["labels"]
                )
                if content and not content.isspace()
            ]
            examples["text"], examples["embeddings"], examples["cluster"], examples["labels"] = zip(
                *filtered_data)
            filtered_labels = [labels for labels in examples["labels"]]
        else:
            filtered_data = [content for content in examples["text"]
                             if content and not content.isspace()]
            examples["text"] = filtered_data

        tokenized_examples = tokenizer(
            examples["text"],
            padding=padding,
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True,
        )

        if "labels" in examples.keys():
            tokenized_examples["labels"] = filtered_labels

        return tokenized_examples

    tokenized_datasets = {}
    dataset_subsets = ["train", "dev", "test"]
    num_cpus = len(os.sched_getaffinity(0))
    num_shards = 8

    for subset in dataset_subsets:
        tokenized_datasets_ls = []
        for shard_i in range(num_shards):
            preprocessed_cache_parent = preprocessed_cache / subset
            preprocessed_cache_parent.mkdir(exist_ok=True)
            preprocessed_cache_i = preprocessed_cache_parent / \
                f"shard_{shard_i}"

            if not preprocessed_cache_i.exists():
                raw_datasets_shard = raw_datasets[subset].shard(
                    num_shards=num_shards, index=shard_i).flatten_indices()
                logger.info(f"Processing {subset} shard {shard_i}")
                tokenized_datasets_i = raw_datasets_shard.map(
                    tokenize_function,
                    batched=True,
                    batch_size=128,
                    num_proc=num_cpus // 2,
                    remove_columns=raw_datasets_shard.column_names,
                    desc="Running tokenizer on dataset line_by_line",
                )
                logger.info(f"Saving {subset} shard {shard_i} to disk")
                tokenized_datasets_i.save_to_disk(str(preprocessed_cache_i))

        for shard_i in range(num_shards):
            assert preprocessed_cache_i.exists()
            tokenized_datasets_i = load_from_disk(str(preprocessed_cache_i))
            if subset == "train":
                for labels in tokenized_datasets_i["labels"]:
                    metadatas.update(labels)
            tokenized_datasets_ls.append(tokenized_datasets_i)
        tokenized_subset = concatenate_datasets(tokenized_datasets_ls)
        tokenized_datasets[subset] = tokenized_subset

    return tokenized_datasets["train"], tokenized_datasets["dev"], tokenized_datasets["test"], metadatas


def get_model(args, load_model=True):
    if args.pipeline_step == 'preprocess':
        load_model = False

    if args.model_name_or_path and not args.from_scratch:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(args.config_dir)
        if args.model_name_or_path.endswith("gpt2"):
            tokenizer.pad_token = tokenizer.eos_token

        if load_model:
            config = AutoConfig.from_pretrained(args.model_name_or_path)

            if args.model_name_or_path.endswith("bert-base-uncased"):
                model = BertForMaskedLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config
                )
            elif args.model_name_or_path.endswith("gpt2"):
                model = GPT2LMHeadModel.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config
                )
            elif args.model_name_or_path.endswith("bart-base"):
                model = BartForConditionalGeneration.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config
                )
    elif args.model_name_or_path and args.from_scratch:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if args.model_name_or_path.endswith("gpt2"):
            tokenizer.pad_token = tokenizer.eos_token

        if load_model:
            if args.model_name_or_path.endswith("bert-base-uncased"):
                model = BertForMaskedLM(config)
            elif args.model_name_or_path.endswith("gpt2"):
                model = GPT2LMHeadModel(config)
            elif args.model_name_or_path.endswith("bart-base"):
                model = BartForConditionalGeneration(config)
    else:
        config = AutoConfig.from_pretrained(args.config_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.config_dir)
        if args.model_name_or_path.endswith("gpt2"):
            tokenizer.pad_token = tokenizer.eos_token

        if load_model:
            if args.model_name_or_path.endswith("bert-base-uncased"):
                model = BertForMaskedLM(config)
            elif args.model_name_or_path.endswith("gpt2"):
                model = GPT2LMHeadModel(config)
            elif args.model_name_or_path.endswith("bart-base"):
                model = BartForConditionalGeneration(config)

    if not load_model:
        model = None

    return tokenizer, model


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    os.environ['NVIDIA_VISIBLE_DEVICES'] = args.cuda_devices

    preprocessed_cache = Path(
        args.dataset_dir) / 'preprocessed_cache' / args.train_dataset_dir.split("/")[-1]
    metadata_save_path = os.path.join(args.train_dataset_dir, "metadata.json")

    if args.pipeline_step == 'preprocess':
        os.makedirs(preprocessed_cache, exist_ok=True)
        logger = get_logger(args)
        tokenizer, model = get_model(args, load_model=False)

        raw_dataset = get_dataset(args)

        try:
            _, _, _, metadatas = preprocess(
                args, raw_dataset, tokenizer, logger, preprocessed_cache)
            with open(metadata_save_path, 'w', encoding='utf-8') as f:
                json_str = json.dumps(list(metadatas))
                f.write(json_str)
        except Exception as e:
            logger.error(traceback.format_exc())

    elif args.pipeline_step == 'pretrain':
        accelerator = Accelerator()
        args.device = accelerator.device
        logger = get_logger(args, accelerator)
        tokenizer, model = get_model(args)
        if accelerator.is_main_process:
            wandb.init(project="PT-Reweighting", name=args.task_name)

        # load dataset
        logger.info("Start loading datasets")

        metadatas = []
        with open(metadata_save_path, 'r', encoding='utf-8') as f:
            metadatas = json.loads(f.read())

        raw_dataset = None
        train_dataset, eval_dataset, test_dataset, _ = preprocess(
            args, raw_dataset, tokenizer, logger, preprocessed_cache=preprocessed_cache)

        if args.model_name_or_path.endswith("bert-base-uncased"):
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm_probability=args.mlm_probability)
        elif args.model_name_or_path.endswith("gpt2"):
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False)
        elif args.model_name_or_path.endswith("bart-base"):
            data_collator = DataCollatorForDenoisingTasks(tokenizer=tokenizer)

        # 自定义 collate 函数
        def collate_fn_train(batch):
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False)

            input_batch = [{'input_ids': item['input_ids'],
                            'attention_mask': item['attention_mask'],
                            'special_tokens_mask': item['special_tokens_mask']} for item in batch]
            metadata = [item.pop('labels') for item in batch]

            # 使用 DataCollatorForLanguageModeling 处理 input_ids 和 attention_mask
            collated_batch = data_collator(input_batch)

            return {"collated_batch": collated_batch, "metadata": metadata}

        def collate_fn_test(batch):
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=False)

            input_batch = [{'input_ids': item['input_ids'],
                            'attention_mask': item['attention_mask'],
                            'special_tokens_mask': item['special_tokens_mask']} for item in batch]

            # 使用 DataCollatorForLanguageModeling 处理 input_ids 和 attention_mask
            collated_batch = data_collator(input_batch)
            return collated_batch

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=collate_fn_train, batch_size=args.per_device_train_batch_size, num_workers=0)
        test_dataloader = DataLoader(
            test_dataset, collate_fn=collate_fn_test, batch_size=args.per_device_eval_batch_size, num_workers=0)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with `accelerator`.
        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None or args.max_train_steps == 0:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(
                args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(args.num_warmup_steps),
            num_training_steps=args.max_train_steps,
        )

        trainer = PretrainTrainer(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            train_dataloader=train_dataloader,
            eval_dataloader=test_dataloader,
            logger=logger,
            accelerator=accelerator,
            from_checkpoint=args.from_ckpt,
            tokenizer=tokenizer,
            max_grad_norm=args.max_grad_norm,
            metadatas=metadatas
        )
        trainer.train()


if __name__ == "__main__":
    main()
