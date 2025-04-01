import json
import re
import ollama


template = """
Task: 
Please generate 3 abstract labels with one word based on the following list of 10 keywords.
The labels should be general and as abstract as possible, aiming to cover the main topics and categories.

You must response with the following format, and don't response anything else:
[Label1], [Label2]

Examples:
[Technology], [Health], [Arts], [Science]

Keyword list:
{}
"""

keyword_path = "/path/to/label1"
save_path = "/path/to/save_label2"
with open(keyword_path, 'r', encoding='utf-8') as f_read:
    with open(save_path, 'a', encoding='utf-8') as f_write:
        for line in f_read.readlines():
            cluster = json.loads(line)
            keywords = ",".join(cluster['label_1'])
            formatted_text = template.format(keywords)
            try:
                response = ollama.chat(model='llama3:70b',
                                       messages=[{'role': 'user', 'content': formatted_text}])['message']['content']
                print(response)
                labels = re.findall(r'\[(.*?)\]', response)
                print(labels)
                cluster["label_2"] = labels
                json_str = json.dumps(cluster) + '\n'

                f_write.write(json_str)
                f_write.flush()
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
