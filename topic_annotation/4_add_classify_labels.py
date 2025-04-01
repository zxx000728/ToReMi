import os
from datasets import load_from_disk

CATEGORIES = (
    "Academic_disciplines",
    "Business",
    "Communication",
    "Concepts",
    "Culture",
    "Economy",
    "Education",
    "Energy",
    "Engineering",
    "Entertainment",
    "Entities",
    "Ethics",
    "Food_and_drink",
    "Geography",
    "Government",
    "Health",
    "History",
    "Human_behavior",
    "Humanities",
    "Information",
    "Internet",
    "Knowledge",
    "Language",
    "Law",
    "Life",
    "Lists",
    "Literature",
    "Mass_media",
    "Mathematics",
    "Military",
    "Nature",
    "People",
    "Philosophy",
    "Politics",
    "Religion",
    "Science",
    "Society",
    "Sports",
    "Technology",
    "Time",
    "Universe",
)

num_cpus = len(os.sched_getaffinity(0))
cluster2labels = {}
reverse_label_mapping = {}


def load_labels(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                line = eval(line)
                labels = set()
                for label in line['labels']:
                    if label in CATEGORIES:
                        labels.add(label)
                cluster2labels[line['cluster_id']] = list(labels)


def add_labels(batch):
    original_labels = [cluster2labels.get(
        str(cluster_id), []) for cluster_id in batch["cluster"]]
    batch["labels"] = original_labels
    return batch


# load dataset
data_path = "/path/to/clustered_data"
dataset = load_from_disk(data_path)
# load labels
keywords_path = "/path/to/cluster_labels"
load_labels(keywords_path)
# add labels
dataset = dataset.map(add_labels, batched=True,
                      batch_size=512, num_proc=num_cpus // 2)
# save dataset
save_path = "/path/to/save_data"
os.makedirs(save_path, exist_ok=True)
dataset.save_to_disk(save_path)
print(dataset[0])
