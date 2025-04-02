
from datasets import load_dataset
dataset = load_dataset("allenai/dolma", name="v1_5-sample", split="train", cache_dir="../path/to/original_dolma")