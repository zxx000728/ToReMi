import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from datasets import load_from_disk
from FlagEmbedding import BGEM3FlagModel


def get_embeddings(raw_datasets, model, save_dir):

    def encode_function(examples):
        examples["text"] = [content for content in examples["text"]
                            if content and content.strip()]
        embeddings = model.encode(examples["text"], batch_size=128, max_length=1024)[
            'dense_vecs']
        examples["embeddings"] = embeddings
        return examples

    os.makedirs(save_dir, exist_ok=True)
    encoded_datasets = raw_datasets.map(
        encode_function,
        batched=True,
        batch_size=128
    )
    encoded_datasets.save_to_disk(save_dir)

    return encoded_datasets


def get_clusters(encoded_dataset, num_clusters, mini_batch_size, save_dir):

    def extract_embeddings(batch):
        batch['embeddings'] = np.array(batch['embeddings'])
        return batch

    print("Start getting embeddings")
    encoded_dataset = encoded_dataset.map(extract_embeddings, batched=True)
    print("Vstacking")
    embeddings = np.vstack(encoded_dataset['embeddings'])

    print("Start clustering")
    minibatch_kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, random_state=0, batch_size=mini_batch_size)
    minibatch_kmeans.fit(embeddings)

    labels = minibatch_kmeans.labels_

    def add_cluster_labels(batch, indices):
        batch["cluster"] = labels[indices]
        return batch

    clustered_dataset = encoded_dataset.map(
        add_cluster_labels, batched=True, with_indices=True)
    clustered_dataset.save_to_disk(save_dir)

    return clustered_dataset


def main():
    data_path = f"/path/to/sample_data"
    model_name = "BAAI/bge-m3"
    embedding_save_dir = f"/path/to/save_embedding_data"

    # encoding
    dataset = load_from_disk(data_path)
    dataset = dataset.remove_columns("effective_tokens")
    model = BGEM3FlagModel(model_name, use_fp16=True)
    encoded_dataset = get_embeddings(dataset, model, embedding_save_dir)
    print(encoded_dataset[0])
    print(len(encoded_dataset))

    # clustering
    num_clusters = 10000
    mini_batch_size = 100000
    cluster_save_dir = f"/path/to/save_clustered_data"
    clustered_dataset = get_clusters(
        encoded_dataset, num_clusters, mini_batch_size, cluster_save_dir)
    print(clustered_dataset[0])


if __name__ == '__main__':
    main()
