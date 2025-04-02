import os
from datasets import load_from_disk
import json
import argparse

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


def parse_args():
    """Configure command line arguments for label integration"""
    parser = argparse.ArgumentParser(description='Add Topic Labels to Dataset')
    parser.add_argument('--cluster_dir', type=str, required=True,
                      help='Path to clustered dataset directory')
    parser.add_argument('--topic_dir', type=str, required=True,
                      help='Path to cluster labels JSONL file')
    parser.add_argument('--save_final_corpora_dir', type=str, required=True,
                      help='Output directory for labeled dataset')
    parser.add_argument('--batch_size', type=int, default=512,
                      help='Processing batch size for dataset mapping')
    return parser.parse_args()


def load_labels(path, cluster2labels):
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
    return cluster2labels

def add_labels(batch):
    original_labels = [cluster2labels.get(
        str(cluster_id), []) for cluster_id in batch["cluster"]]
    batch["labels"] = original_labels
    return batch

if __name__ == '__main__':

    args = parse_args()
    num_cpus = len(os.sched_getaffinity(0))
    cluster2labels = {}

    # load dataset
    cluster_dir = args.cluster_dir
    dataset = load_from_disk(cluster_dir)
    # load labels
    topic_dir = args.topic_dir
    cluster2labels = load_labels(topic_dir, cluster2labels)
    # add labels
    dataset = dataset.map(add_labels, batched=True,
                        batch_size=args.batch_size, num_proc=num_cpus // 2)
    # save dataset
    save_final_corpora_dir = args.save_final_corpora_dir
    os.makedirs(save_final_corpora_dir, exist_ok=True)
    dataset.save_to_disk(save_final_corpora_dir)
    print(dataset[0])
