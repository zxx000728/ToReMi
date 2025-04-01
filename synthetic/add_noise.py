import os
import random
from datasets import load_from_disk


def shuffle_all_text(text):
    """
    随机打乱整个文本
    """
    text_list = list(text)
    random.shuffle(text_list)
    return ''.join(text_list)


def shuffle_all_text_technology_samples_batch(batch):
    # 打乱整个text
    batch["text"] = [
        shuffle_all_text(text) if "Technology" in labels else text
        for text, labels in zip(batch["text"], batch["labels"])
    ]
    return batch


data_path = "/path/to/data"
dataset = load_from_disk(data_path)
num_cpus = len(os.sched_getaffinity(0))

noisy_dataset = dataset.map(shuffle_all_text_technology_samples_batch,
                            batched=True, batch_size=512, num_proc=num_cpus // 2)
save_path = "/path/to/save"
noisy_dataset.save_to_disk(save_path)

# # check results
# for sample in noisy_dataset:
#     if "Technology" in sample["labels"]:
#         print(sample)
#         break
