from transformers import GPT2ForSequenceClassification, GPT2Tokenizer

model_path = "/path/to/model"
save_path = "/path/to/classification-one"

model = GPT2ForSequenceClassification.from_pretrained(model_path, num_labels=1)
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

# Define a padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Save the model and tokenizer in a directory
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
