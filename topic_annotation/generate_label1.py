import json
import re
import ollama


template = """
Task: 
Please generate 10 abstract labels with one word based on the following list of 100 keywords.
The labels should be general and as abstract as possible, aiming to cover the main topics and categories.

You must response with the following format, and don't response anything else:
[Label1], [Label2]

Examples:
[Technology], [Health], [Arts], [Science]

Keyword list:
{}
"""

keyword_path = "/path/to/cluster_keywords"
with open(keyword_path, 'r', encoding='utf-8') as f:
    keywords_dict = json.load(f)

save_path = "/path/to/save_label1"
with open(save_path, 'a', encoding='utf-8') as f:
    for cluster_id, keyword_list in keywords_dict.items():
        new_dict = {"cluster_id": cluster_id, "keywords": keyword_list}

        keywords = ",".join(keyword_list[:100])
        formatted_text = template.format(keywords)
        try:
            response = ollama.chat(model='llama3:70b',
                                   messages=[{'role': 'user', 'content': formatted_text}])['message']['content']
            print(response)
            labels = re.findall(r'\[(.*?)\]', response)
            print(labels)
            new_dict["label_1"] = labels
            json_str = json.dumps(new_dict) + '\n'
            f.write(json_str)
            f.flush()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
