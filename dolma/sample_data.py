import os
import multiprocessing
import argparse
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Sample Dolma dataset with configurable parameters")
    parser.add_argument('--dolma_dir', type=str, required=True, 
                      help='Root directory containing original Dolma datasets')
    parser.add_argument('--save_dir', type=str, required=True,
                      help='Output directory for sampled datasets')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2',
                      help='Pretrained tokenizer name from Hugging Face Hub')
    parser.add_argument('--sample_token_length', type=int, default=1024,
                      help='Length of token chunks for text splitting')
    parser.add_argument('--total_tokens_targets', nargs='+', type=int, 
                      default=[2600000000, 7200000000, 15600000000, 30000000000],
                      help='Space-separated list of total token targets for each sampling stage')
    parser.add_argument('--output_paths', nargs='+', type=str,
                      default=["data_2_6B", "data_7_2B", "data_15_6B", "data_30B"],
                      help='Space-separated output directory names for each sampling stage')
    parser.add_argument('--sampling_ratios', type=str, 
                      default='books:0.002,code:0.138,ref:0.025,web:0.835',
                      help='Comma-separated category:ratio pairs for sampling')
    parser.add_argument('--num_cpus', type=int, default= os.cpu_count(),
                      help='Number of CPU cores to use (default: all available)')
    return parser.parse_args()


def chunk_and_count_text(examples):
    """Split text into fixed-length token chunks and count effective tokens
    Args:
        examples: Input batch containing text data
    Returns:
        Dictionary with chunked texts and corresponding token counts
    """
    tokenized_texts = tokenizer(
        examples['text'], truncation=False, add_special_tokens=False)['input_ids']
    chunked_texts = []
    effective_token_counts = []

    for tokens in tokenized_texts:
        for i in range(0, len(tokens), sample_token_length):
            chunk = tokens[i:i + sample_token_length]
            effective_tokens = len(chunk)

            chunked_texts.append(tokenizer.decode(
                chunk, skip_special_tokens=True))
            effective_token_counts.append(effective_tokens)

    return {'text': chunked_texts, 'effective_tokens': effective_token_counts}


# 最终统计 token 数
def count_tokens(examples):
    """Count tokens in text samples for verification"""
    tokenized = tokenizer(examples['text'], add_special_tokens=False)
    return {'tokens': [len(input_ids) for input_ids in tokenized['input_ids']]}


if __name__ == '__main__':
    args = parse_args()
    
    dolma_dir = args.dolma_dir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Process sampling ratios
    sampling_ratios = dict(pair.split(':') for pair in args.sampling_ratios.split(','))
    sampling_ratios = {k: float(v) for k, v in sampling_ratios.items()}
    
    # Configure file paths
    json_files = {
        "books": f"{dolma_dir}/books_data.json",
        "code": f"{dolma_dir}/code_data.json",
        "ref": f"{dolma_dir}/ref_data.json",
        "web": f"{dolma_dir}/web_data.json"
    }
    
    # Initialize processing resources
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    num_cpus = args.num_cpus

    print("Loading datasets...")
    datasets = {}
    for source, file_path in json_files.items():
        datasets[source] = load_dataset(
            "json", data_files=file_path, split="train")

    print("Processing datasets...")
    chunked_datasets = {}
    for source, dataset in datasets.items():
        print(f"Processing {source} using {num_cpus // 2} processes...")
        chunked_datasets[source] = dataset.map(
            chunk_and_count_text,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=num_cpus // 2
        )

    print("Sampling datasets...")
    for total_tokens_required, output_path in zip(total_tokens_targets, output_paths):
        print(f"Sampling {total_tokens_required} dataset")
        print("Computing sample num")
        tokens_per_source = {src: int(ratio * total_tokens_required)
                            for src, ratio in sampling_ratios.items()}
        samples_per_source = {}
        for source, dataset in chunked_datasets.items():
            total_effective_tokens = sum(dataset['effective_tokens'])
            samples_per_source[source] = int(
                tokens_per_source[source] / total_effective_tokens * len(dataset))

        print("Selecting samples")
        # 按比例抽样
        sampled_datasets = []
        last_remain_num = 0
        for source, dataset in chunked_datasets.items():
            num_samples = samples_per_source[source]  # 当前source需要sample的数据量
            num_samples += last_remain_num
            num_dataset = len(dataset)  # 当前dataset的数据量
            if num_dataset <= num_samples:  # 当前dataset不够，到下一个dataset再继续采样
                sampled_dataset = dataset
                last_remain_num = num_samples - num_dataset  # 剩下要采样的数量
            else:
                sampled_dataset = dataset.shuffle(
                    seed=42).select(range(num_samples))
                last_remain_num = 0
            print(last_remain_num)
            sampled_datasets.append(sampled_dataset)

        print("Concatenating dataset")
        # 合并当前阶段数据集
        stage_dataset = concatenate_datasets(sampled_datasets)

        # 保存当前阶段数据集
        print(f"Saving {total_tokens_required} dataset to {output_path}")
        output_path = os.path.join(save_dir, output_path)
        stage_dataset.save_to_disk(output_path)

        print("Checking token num")
        # check token num
        token_counts = stage_dataset.map(
            count_tokens,
            batched=True,
            num_proc=num_cpus // 2
        )
        total_tokens = sum(token_counts['tokens'])
        print(f"Final total tokens: {total_tokens}")

        print("Checking sampled data")
        # check sample data
        sampled_examples = stage_dataset.shuffle(seed=42).select(range(5))
        for i, example in enumerate(sampled_examples):
            print(f"Sample {i+1}: {example}")


    print("All datasets saved successfully!")
