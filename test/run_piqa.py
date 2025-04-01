import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json


MODEL_PATH = "/path/to/model"
GPU_ID = 0
BATCH_SIZE = 4

VALID_JSON = "/path/to/piqa/valid.jsonl"
VALID_LABELS = "/path/to/piqa/valid-labels.lst"


def load_local_dataset():
    """加载本地数据集文件"""
    # 读取 jsonl 文件
    data = []
    with open(VALID_JSON, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    # 读取标签文件
    labels = []
    with open(VALID_LABELS, 'r', encoding='utf-8') as f:
        for line in f:
            labels.append(int(line.strip()))

    # 确保数据和标签长度匹配
    assert len(data) == len(
        labels), f"Data length ({len(data)}) doesn't match labels length ({len(labels)})"

    # 构建数据集
    dataset = []
    for item, label in zip(data, labels):
        dataset.append({
            'goal': item['goal'],
            'sol1': item['sol1'],
            'sol2': item['sol2'],
            'label': label
        })

    print(f"Loaded {len(dataset)} examples from local files")
    return dataset


def get_sequence_score(model, tokenizer, input_text, device):
    """计算序列的困惑度分数"""
    inputs = tokenizer(input_text, return_tensors="pt",
                       truncation=True, max_length=1024).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # 移除padding token的影响
        attention_mask = inputs['attention_mask']
        logits = logits[attention_mask.bool()]

        # 计算每个位置的损失
        shift_logits = logits[:-1]
        shift_labels = inputs['input_ids'][0][1:]

        # 计算交叉熵损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits, shift_labels)

        # 返回平均损失（越小越好）
        return -token_losses.mean().item()


def evaluate_piqa(model, tokenizer, dataset, device, batch_size=4):
    model.eval()
    correct = 0
    total = 0

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]

        scores = []
        with torch.no_grad():
            for item in batch:
                goal = item['goal']
                s1 = item['sol1']
                s2 = item['sol2']

                # 构建完整的输入文本
                input1 = f"{goal} {s1}"
                input2 = f"{goal} {s2}"

                # 计算两个选项的分数
                score1 = get_sequence_score(model, tokenizer, input1, device)
                score2 = get_sequence_score(model, tokenizer, input2, device)

                # 选择困惑度更低的选项
                pred = 1 if score2 > score1 else 0
                scores.append(pred)

        # 获取当前批次的标签
        labels = [item['label'] for item in batch]
        # 计算准确率
        batch_correct = sum(1 for pred, label in zip(
            scores, labels) if pred == label)
        correct += batch_correct
        total += len(scores)

    return correct / total


def main():
    device = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型和分词器
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)

    # 加载PIQA数据集
    print("Loading local dataset files...")
    dataset = load_local_dataset()

    # 评估模型
    print("Starting evaluation...")
    accuracy = evaluate_piqa(model, tokenizer, dataset, device, BATCH_SIZE)
    print(f"Final accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
