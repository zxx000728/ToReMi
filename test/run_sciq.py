import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score

# 配置参数
CONFIG = {
    'tokenizer_path': 'gpt2',
    'model_path': '/path/to/model',
    'gpu_id': 0,
    'max_length': 1024,
}

# 1-shot
EXAMPLE = {
    'support': "Without Coriolis Effect the global winds would blow north to south or south to north. But Coriolis makes them blow northeast to southwest or the reverse in the Northern Hemisphere. The winds blow northwest to southeast or the reverse in the southern hemisphere.",
    'question': "What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?",
    'correct_answer': "coriolis effect",
    'distractor1': "muon effect",
    'distractor2': "centrifugal effect",
    'distractor3': "tropical effect"
}


def format_qa_pair(support, question, options, answer_letter=None):
    """Format a single QA pair"""
    text = f"Context: {support}\n\n"
    text += f"Question: {question}\n\n"
    text += "Options:\n"
    for i, opt in enumerate(['A', 'B', 'C', 'D']):
        text += f"{opt}. {options[i]}\n"
    text += "\nAnswer: "
    if answer_letter:
        text += f"{answer_letter}\n\n"
    return text


def format_prompt(question, support, options, include_shot=False):
    """Format the prompt with optional 1-shot example"""
    prompt = ""

    if include_shot:
        # 构建示例的选项列表，确保正确答案在A位置
        example_options = [
            EXAMPLE['correct_answer'],
            EXAMPLE['distractor1'],
            EXAMPLE['distractor2'],
            EXAMPLE['distractor3']
        ]
        # 添加示例
        prompt += format_qa_pair(
            EXAMPLE['support'],
            EXAMPLE['question'],
            example_options,
            'A'  # 示例中正确答案总是A
        )

    # 添加当前问题
    prompt += format_qa_pair(support, question, options)
    return prompt


def get_prediction(model, tokenizer, prompt, device):
    """使用模型计算每个选项的概率并返回概率最高的选项"""
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=CONFIG['max_length'])
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # 获取最后一个token的输出
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]

        # 获取A、B、C、D对应的token_id
        option_tokens = ['A', 'B', 'C', 'D']
        option_ids = [tokenizer.convert_tokens_to_ids(
            t) for t in option_tokens]

        # 计算这些token的概率
        option_logits = logits[:, option_ids]
        probs = torch.softmax(option_logits, dim=-1)[0]

        # 获取最高概率的选项
        max_idx = probs.argmax().item()
        max_prob = probs[max_idx].item()

        all_probs = {opt: prob.item()
                     for opt, prob in zip(option_tokens, probs)}

        return option_tokens[max_idx], max_prob, all_probs


def main():
    device = f"cuda:{CONFIG['gpu_id']}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer_path'])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG['model_path'],
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    print("Loading SciQ dataset...")
    dataset = load_dataset("allenai/sciq")
    test_dataset = dataset["test"]

    correct = 0
    total = 0
    predictions = []
    labels = []
    low_confidence_count = 0  # 记录低置信度预测的数量
    confidence_threshold = 0.5  # 置信度阈值

    # 记录每个选项的平均概率
    prob_stats = {'A': [], 'B': [], 'C': [], 'D': []}

    print("Starting evaluation...")
    for item in tqdm(test_dataset):
        # Format options
        options = [
            item['correct_answer'],
            item['distractor1'],
            item['distractor2'],
            item['distractor3']
        ]
        # Randomly shuffle options and keep track of correct answer
        correct_idx = 0  # Initially, correct answer is at index 0
        np.random.shuffle(options)
        for i, opt in enumerate(options):
            if opt == item['correct_answer']:
                correct_idx = i
                break

        prompt = format_prompt(
            item['question'],
            item['support'],
            options,
            include_shot=False  # 包含1-shot示例
        )

        # Get model prediction with probabilities
        pred, confidence, all_probs = get_prediction(
            model, tokenizer, prompt, device)

        # 记录所有概率
        for opt, prob in all_probs.items():
            prob_stats[opt].append(prob)

        # Convert prediction to index (A->0, B->1, etc.)
        pred_idx = ord(pred) - ord('A')

        # 记录低置信度预测
        if confidence < confidence_threshold:
            low_confidence_count += 1
            if total % 10 == 0:  # 每10个样本打印一次低置信度预测的详情
                print(f"\nLow confidence prediction:")
                print(f"Question: {item['question']}")
                print(f"Predicted: {pred} (confidence: {confidence:.4f})")
                print(f"All probabilities: {all_probs}")

        predictions.append(pred_idx)
        labels.append(correct_idx)

        if pred_idx == correct_idx:
            correct += 1
        total += 1

        # Print running accuracy
        if total % 10 == 0:
            print(f"\nRunning accuracy: {correct/total:.4f}")
            print(f"Low confidence predictions so far: {low_confidence_count}")

    # Calculate final metrics
    final_accuracy = accuracy_score(labels, predictions)

    # 计算每个选项的平均概率
    avg_probs = {opt: np.mean(probs) for opt, probs in prob_stats.items()}

    print("\nEvaluation Results:")
    print(f"Total examples: {total}")
    print(f"Correct predictions: {correct}")
    print(
        f"Low confidence predictions: {low_confidence_count} ({low_confidence_count/total*100:.2f}%)")
    print(f"Final accuracy: {final_accuracy:.4f}")
    print("\nAverage probabilities for each option:")
    for opt, avg_prob in avg_probs.items():
        print(f"Option {opt}: {avg_prob:.4f}")


if __name__ == "__main__":
    main()
