import json
import os
import ollama
import argparse
from typing import List

def parse_args():
    """Configure command line arguments for topic classification"""
    parser = argparse.ArgumentParser(description='Cluster Topic Classification using LLMs')
    parser.add_argument('--keywords_dir', type=str, required=True,
                      help='Path to cluster keywords JSON file')
    parser.add_argument('--save_topic_dir', type=str, required=True,
                      help='Output directory for classification results')
    parser.add_argument('--model_name', type=str, default='llama3:70b',
                      help='Ollama model name for classification')
    parser.add_argument('--max_retry', type=int, default=5,
                      help='Max retries for API failures')
    return parser.parse_args()

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

# template = """
# Task: Select one to three most related topic labels based on the given keywords.

# [Keywords]
# {}

# [Topic labels]
# {}

# Based on the given keywords, please select one to three most relevant labels from the provided topic labels.
# Ensure that the selected labels best capture the primary concepts and topics represented by the keywords.

# Note:
# 1. The selected labels must be from the given topic labels!
# 2. Don't respond any reasoning process or explanations!

# You must respond with the following format, and don't respond anything else:
# ### Labels: Label1, Label2

# Examples:
# ### Labels: xxxx, xxxx, ...
# """

def generate_prompt(keywords: List[str], categories: List[str]) -> str:
    return f"""
Task: Select one to three most related topic labels based on the given keywords.

[Keywords]
{", ".join(keywords[:50])}

[Topic labels]
{", ".join(categories)}

Based on the given keywords, please select one to three most relevant labels from the provided topic labels.
Ensure that the selected labels best capture the primary concepts and topics represented by the keywords.

Note:
1. The selected labels must be from the given topic labels!
2. Don't respond any reasoning process or explanations!

You must respond with the following format, and don't respond anything else:
### Labels: Label1, Label2

Examples:
### Labels: xxxx, xxxx, ...
"""


if __name__ == '__main__':
    args = parse_args()
    keyword_path = args.keyword_path
    save_topic_dir = args.save_topic_dir
    model_name = args.model_name
    MAX_RETRY = args.max_retry

    with open(keyword_path, 'r', encoding='utf-8') as f:
        keywords_dict = json.load(f)

    os.makedirs(save_topic_dir, exist_ok=True)
    save_path = os.path.join(save_topic_dir, "clsuter_label.json")
    error_save_path = os.path.join(save_topic_dir, "error.json")


    with open(save_path, 'a', encoding='utf-8') as f:
        for cluster_id, keyword_list in keywords_dict.items():
            new_dict = {"cluster_id": cluster_id, "keywords": keyword_list}

            formatted_text = generate_prompt(keywords, CATEGORIES)

            curr_retry = 0
            is_valid = True

            while curr_retry < MAX_RETRY:
                try:
                    response = ollama.chat(model=model_name,
                                        messages=[{'role': 'user', 'content': formatted_text}])['message']['content']
                    print(response)
                    labels = response.split('### Labels:')[-1].strip().split(',')
                    labels = [label.strip() for label in labels]
                    print(labels)
                    if not labels:
                        is_valid = False
                    for label in labels:
                        if label not in CATEGORIES:
                            is_valid = False
                            break
                    if is_valid:
                        new_dict["labels"] = labels
                        json_str = json.dumps(new_dict, ensure_ascii=False) + '\n'
                        f.write(json_str)
                        f.flush()
                        break
                    else:
                        curr_retry += 1
                        is_valid = True
                        if curr_retry == MAX_RETRY:
                            raise Exception("Invalid response!")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                    with open(error_save_path, 'a', encoding='utf-8') as f2:
                        json_str = json.dumps(new_dict, ensure_ascii=False) + '\n'
                        f2.write(json_str)
