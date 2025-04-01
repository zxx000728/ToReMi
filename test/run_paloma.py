import torch
import math
import numpy as np
from datasets import load_from_disk
from transformers import GPT2Tokenizer, GPT2LMHeadModel, DataCollatorForLanguageModeling
from torch.utils.data.dataloader import DataLoader


model_path = "/path/to/model"
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

test_subsets = ["wikitext_103", "m2d2_s2orc_unsplit", "manosphere_meta_sep", "ptb",
                "4chan_meta_sep", "c4_en", "m2d2_wikipedia_unsplit", "twitterAAE_HELM_fixed",
                "c4_100_domains", "mc4", "gab", "redpajama"]
test_datasets = [load_from_disk(
    f"/path/to/paloma/{subset}") for subset in test_subsets]

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
test_dataloaders = {name: DataLoader(dataset, collate_fn=data_collator,
                                     batch_size=256, num_workers=0
                                     ) for name, dataset in zip(test_subsets, test_datasets)}


def calculate_loss_and_accuracy_and_ppl(outputs, labels):
    loss = outputs.loss
    logits = outputs.logits

    _, preds = logits.max(dim=-1)
    not_ignore = labels.ne(-100)
    num_targets = not_ignore.long().sum().item()

    correct = (labels == preds) & not_ignore
    correct = correct.float().sum()

    accuracy = correct / num_targets
    ppl = math.exp(loss)

    return loss, accuracy, ppl


def evaluate(model, eval_dataloader):
    total_valid_loss_list, total_valid_acc_list, total_valid_ppl_list = [], [], []
    model.eval()
    res = {}
    with torch.no_grad():
        for test_name, eval_dataloader in eval_dataloader.items():
            sub_valid_loss_list, sub_valid_acc_list, sub_valid_ppl_list = [], [], []
            for eval_step, batch in enumerate(eval_dataloader):
                outputs = model(**batch, return_dict=True)
                labels = batch["labels"]
                loss, acc, ppl = calculate_loss_and_accuracy_and_ppl(
                    outputs, labels)
                # add to total list
                total_valid_loss_list.append(float(loss))
                total_valid_acc_list.append(float(acc))
                total_valid_ppl_list.append(float(ppl))
                # add to sub list
                sub_valid_loss_list.append(float(loss))
                sub_valid_acc_list.append(float(acc))
                sub_valid_ppl_list.append(float(ppl))

            sub_valid_mean_ppl = np.mean(sub_valid_ppl_list)
            res[test_name] = sub_valid_mean_ppl

        valid_mean_loss = np.mean(total_valid_loss_list)
        valid_mean_acc = np.mean(total_valid_acc_list)
        valid_mean_ppl = np.mean(total_valid_ppl_list)
        print("Test, average, loss {}, accuracy {}, ppl {}".format(
            valid_mean_loss, valid_mean_acc, valid_mean_ppl
        ))
    print(res)


evaluate(model, test_dataloaders)
