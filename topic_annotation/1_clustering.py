import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from datasets import load_from_disk
import argparse
from FlagEmbedding import BGEM3FlagModel



def parse_args():
    """Parse command line arguments for clustering configuration"""
    parser = argparse.ArgumentParser(description='Text Embedding and Clustering Pipeline')
    parser.add_argument('--corpora_dir', type=str, required=True,
                      help='Path to sampled dataset directory')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-m3',
                      help='Pretrained model name for embeddings')
    parser.add_argument('--save_embedding_dir', type=str, required=True,
                      help='Output directory for storing embeddings')
    parser.add_argument('--save_cluster_dir', type=str, required=True,
                      help='Output directory for clustered results')
    parser.add_argument('--num_clusters', type=int, default=10000,
                      help='Number of clusters for K-means algorithm')
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for embedding generation')
    parser.add_argument('--max_length', type=int, default=1024,
                      help='Max input token length')
    parser.add_argument('--mini_batch_size', type=int, default=100000,
                      help='Batch size for MiniBatchKMeans clustering')
    parser.add_argument('--fp16', action='store_true',
                      help='Use FP16 precision for model inference')
    return parser.parse_args()

def get_embeddings(raw_datasets, model, save_dir, batch_size=128, max_length=1024):
    """Generate text embeddings using specified model
    
    Args:
        raw_datasets: Input dataset containing text samples
        model: Pretrained embedding model
        save_dir: Directory to save processed datasets with embeddings
        batch_size: Processing batch size
        max_length: Max input token length
    
    Returns:
        Dataset with generated embeddings
    """
    def encode_function(examples):
        examples["text"] = [content for content in examples["text"]
                            if content and content.strip()]
        embeddings = model.encode(examples["text"], batch_size=batch_size, max_length=max_length)[
            'dense_vecs']
        examples["embeddings"] = embeddings
        return examples

    os.makedirs(save_dir, exist_ok=True)
    encoded_datasets = raw_datasets.map(
        encode_function,
        batched=True,
        batch_size=batch_size
    )
    encoded_datasets.save_to_disk(save_dir)

    return encoded_datasets


def get_clusters(encoded_dataset, num_clusters, mini_batch_size, save_dir):
    """Perform K-means clustering on text embeddings
    
    Args:
        encoded_dataset: Dataset containing text embeddings
        num_clusters: Number of clusters to create
        mini_batch_size: Batch size for incremental clustering
        save_dir: Directory to save clustered results
    
    Returns:
        Dataset with cluster labels added
    """
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

    args = parse_args()

    corpora_dir = args.corpora_dir
    model_name = args.model_name
    save_embedding_dir = args.save_embedding_dir
    use_fp16 = args.use_fp16
    batch_size = args.batch_size
    max_length = args.max_length

    num_clusters = args.num_clusters
    mini_batch_size = args.mini_batch_size
    save_cluster_dir = args.save_cluster_dir

    # encoding
    dataset = load_from_disk(corpora_dir)
    dataset = dataset.remove_columns("effective_tokens")
    model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
    encoded_dataset = get_embeddings(dataset, model, save_embedding_dir, batch_size, max_length)
    print(encoded_dataset[0])
    print(len(encoded_dataset))

    # clustering
    clustered_dataset = get_clusters(
        encoded_dataset, num_clusters, mini_batch_size, save_cluster_dir)
    print(clustered_dataset[0])


if __name__ == '__main__':
    main()
