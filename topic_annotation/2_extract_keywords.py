import os
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import re
import argparse
import json


def parse_args():
    """Parse command line arguments for keyword extraction"""
    parser = argparse.ArgumentParser(description='Cluster Keyword Extraction')
    parser.add_argument('--cluster_dir', type=str, required=True,
                      help='Path to clustered dataset directory')
    parser.add_argument('--save_keywords_dir', type=str, required=True,
                      help='Directory to save keyword JSON files')
    parser.add_argument('--top_n', type=int, default=100,
                      help='Number of keywords to extract per cluster')
    parser.add_argument('--lang', type=str, default='english',
                      help='Stopword language for text preprocessing')
    parser.add_argument('--no_stopwords', action='store_true',
                      help='Disable stopword filtering')
    return parser.parse_args()


def initialize_stopwords(lang='english'):
    """Initialize NLTK stopwords and ensure resource availability"""
    try:
        nltk.data.find(f'corpora/stopwords/{lang}')
    except LookupError:
        nltk.download('stopwords')
    return set(stopwords.words(lang))


def preprocess(text, stopwords_set=[]):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    if stopwords_set:
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

    args = parse_args()

    cluster_dir = args.cluster_dir
    save_keywords_dir = args.save_keywords_dir
    top_n = args.top_n
    lang = args.lang
    no_stopwords = args.no_stopwords

    if no_stopwords:
        nltk.download('stopwords')
        stop_words = set(stopwords.words(lang))
    else:
        stop_words = []

    dataset = load_from_disk(cluster_dir)

    # Preprocess texts
    dataset = dataset.map(
        lambda example: {'contents': preprocess(example['text'], stop_words)})

    # Organize texts by cluster
    cluster_contents = {}
    for example in dataset:
        cluster_id = example['cluster']
        if cluster_id not in cluster_contents:
            cluster_contents[cluster_id] = []
        if example['contents'].strip():
            cluster_contents[cluster_id].append(example['contents'])

    # Extract and save keywords
    cluster_keywords = {}
    for cluster_id, texts in cluster_contents.items():
        keywords = extract_keywords(texts, top_n)
        cluster_keywords[cluster_id] = keywords

    os.makedirs(save_keywords_dir, exist_ok=True)
    output_file = os.path.join(save_keywords_dir, "keywords.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cluster_keywords, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
