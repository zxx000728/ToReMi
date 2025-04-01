import os
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import re
import json


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


def extract_keywords(texts, top_n=100):
    if not texts:
        return []
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        tfidf_scores = dict(zip(feature_names, tfidf_scores))
        sorted_keywords = sorted(tfidf_scores.items(
        ), key=lambda x: x[1], reverse=True)[:top_n]
        return [word for word, score in sorted_keywords]
    except ValueError:
        return []


def main():
    data_dir = f"/path/to/clustered_data"
    dataset = load_from_disk(data_dir)

    # 去除停用词
    dataset = dataset.map(
        lambda example: {'contents': preprocess(example['text'])})

    # 按簇分组
    cluster_contents = {}
    for example in dataset:
        cluster_id = example['cluster']
        if cluster_id not in cluster_contents:
            cluster_contents[cluster_id] = []
        if example['contents'].strip():
            cluster_contents[cluster_id].append(example['contents'])

    # 提取每个簇的关键词
    cluster_keywords = {}
    for cluster_id, texts in cluster_contents.items():
        keywords = extract_keywords(texts)
        cluster_keywords[cluster_id] = keywords

    # 将结果写入JSON文件
    output_dir = f"/path/to/save_cluster_keywords"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "keywords.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_keywords, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
