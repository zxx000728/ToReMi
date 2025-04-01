import math
import torch
import numpy as np
from numpy.random import permutation, poisson
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class DataCollatorForLanguageModeling:
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(
            self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(
            examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        batch.pop("id", None)

        init_labels = batch.pop("labels", None)
        input_ids = batch["input_ids"].clone()
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                input_ids, special_tokens_mask=special_tokens_mask,
                init_labels=init_labels,
            )
        else:
            batch.pop("labels", None)
        return batch

    def mask_tokens(
            self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None,
            init_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        if init_labels is not None:
            labels[inputs == self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.mask_token)] = init_labels

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


@dataclass
class DataCollatorForDenoisingTasks:
    """Data collator used denoising language modeling task in BART.
    (text-infilling + sentence-permutation data collator)
    The implementation is based on
    https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/fairseq/data/denoising_dataset.py.
    The default parameters is based on BART paper https://arxiv.org/abs/1910.13461.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_ratio: float = 0.3
    # text infilling随机抽样被替换文本(span)的长度从泊松分布(lambda=3)中抽
    poisson_lambda: float = 3.0
    permutate_sentence_ratio: float = 0.0
    pad_to_multiple_of: Optional[int] = None

    # delete_token_id: int = 5

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.eos_token is None:
            raise ValueError

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """Batching, adding whole word mask and permutate sentences

        input_string = "<s> My dog is <mask> </s>"
        decoder_input_string = "</s><s> My dog is cute"
        labels_string = "My dog is cute </s>"

        input_ids = tok(input_string, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids =tok(decoder_input_string, add_special_tokens=False, return_tensors="pt").input_ids
        labels = tok(labels_string, add_special_tokens=False, return_tensors="pt").input_ids

        loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels)[0]

        Args:
            examples (dict): list of examples each examples contains input_ids field
        """

        # Handle dict or lists with proper padding and conversion to tensor.
        # Pad a single encoded input or a batch of encoded inputs up to predefined length
        # or to the max sequence length in the batch.
        batch = self.tokenizer.pad(
            examples, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="np")

        batch["decoder_input_ids"] = torch.tensor(
            self.shift_tokens_right(batch["input_ids"]))
        batch["attention_mask"] = torch.tensor(batch["attention_mask"])
        batch.pop("special_tokens_mask", None)

        do_permutate = False
        if self.permutate_sentence_ratio > 0.0:
            batch["input_ids"] = torch.tensor(
                self.permutate_sentences(batch["input_ids"]))
            do_permutate = True

        if self.mask_ratio:
            batch["input_ids"], batch["labels"] = self.add_whole_word_mask(
                batch["input_ids"], do_permutate)
            batch["input_ids"] = torch.tensor(batch["input_ids"])
            batch["labels"] = torch.tensor(batch["labels"])

        return batch

    def shift_tokens_right(self, inputs):
        """Shift decoder input ids right: https://github.com/huggingface/transformers/issues/7961.
        Examples:
            <s>My dog is cute.</s><s>It loves to play in the park.</s><pad><pad>
            shift to -> </s><s>My dog is cute.</s><s>It loves to play in the park.<pad><pad>
        """

        shifted_inputs = np.roll(inputs, 1, axis=-1)  # 滚动数组，超出最后位置的元素会滚动到第一个位置

        # replace first token with eos token
        shifted_inputs[:, 0] = self.tokenizer.sep_token_id

        # when there's padding, the last eos tokens will not be rotated to first position
        # we'll need to replace it with a padding token

        # replace eos tokens at the end of sequences with pad tokens
        end_with_eos = np.where(
            shifted_inputs[:, -1] == self.tokenizer.sep_token_id)
        shifted_inputs[end_with_eos, -1] = self.tokenizer.pad_token_id

        # find positions where the token is eos and its following token is a padding token
        last_eos_indices = np.where(
            (shifted_inputs[:, :-1] == self.tokenizer.sep_token_id)
            * (shifted_inputs[:, 1:] == self.tokenizer.pad_token_id)
        )

        # replace eos tokens with pad token
        shifted_inputs[last_eos_indices] = self.tokenizer.pad_token_id
        return shifted_inputs

    def permutate_sentences(self, inputs):
        results = inputs.copy()

        full_stops = inputs == self.tokenizer.sep_token_id

        sentence_ends = np.argwhere(full_stops[:, 1:] * ~full_stops[:, :-1])
        sentence_ends[:, 1] += 2
        num_sentences = np.unique(sentence_ends[:, 0], return_counts=True)[1]
        num_to_permute = np.ceil(
            (num_sentences * 2 * self.permutate_sentence_ratio) / 2.0).astype(int)

        sentence_ends = np.split(sentence_ends[:, 1], np.unique(
            sentence_ends[:, 0], return_index=True)[1][1:])

        for i in range(inputs.shape[0]):
            # 产生一个随机序列，选择要重排的句子及顺序
            substitutions = np.random.permutation(num_sentences[i])[
                : num_to_permute[i]]

            ordering = np.arange(0, num_sentences[i])
            # 决定哪句话替换哪句话
            ordering[substitutions] = substitutions[np.random.permutation(
                num_to_permute[i])]

            index = 0
            for j in ordering:
                sentence = inputs[i, (sentence_ends[i][j - 1]
                                      if j > 0 else 0): sentence_ends[i][j]]
                results[i, index: index + sentence.shape[0]] = sentence
                index += sentence.shape[0]
        return results

    def add_whole_word_mask(self, inputs, do_permutate):
        labels = inputs.copy()

        # A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = np.array(special_tokens_mask, dtype=bool)

        # determine how many tokens we need to mask in total
        is_token = ~(
            labels == self.tokenizer.pad_token_id) & ~special_tokens_mask
        num_to_mask = int(
            math.ceil(is_token.astype(float).sum() * self.mask_ratio))
        if num_to_mask == 0:
            return inputs, labels

        # generate a sufficient number of span lengths
        lengths = poisson(lam=self.poisson_lambda, size=(num_to_mask,))
        while np.cumsum(lengths, 0)[-1] < num_to_mask:
            lengths = np.concatenate(
                [lengths, poisson(lam=self.poisson_lambda, size=(num_to_mask,))])

        # remove all spans of length 0
        # Note that BART inserts additional mask tokens where length == 0,
        # which we do not implement for now as it adds additional complexity
        lengths = lengths[lengths > 0]

        # trim to about num_to_mask tokens
        idx = np.argmin(np.abs(np.cumsum(lengths, 0) - num_to_mask)) + 1
        lengths = lengths[: idx + 1]

        # select span start indices
        token_indices = np.argwhere(is_token == 1)
        span_starts = permutation(token_indices.shape[0])[: lengths.shape[0]]

        # prepare mask
        masked_indices = np.array(token_indices[span_starts])
        mask = np.full_like(labels, fill_value=False)

        # mask span start indices
        for mi in masked_indices:
            mask[tuple(mi)] = True
        lengths -= 1

        # fill up spans
        max_index = labels.shape[1] - 1
        remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)
        while np.any(remaining):
            masked_indices[remaining, 1] += 1
            for mi in masked_indices:
                mask[tuple(mi)] = True
            lengths -= 1
            remaining = (lengths > 0) & (masked_indices[:, 1] < max_index)

        # place the mask tokens
        mask[np.where(special_tokens_mask)] = False
        inputs[np.where(mask)] = self.tokenizer.mask_token_id

        if not do_permutate:
            # labels[np.where(mask)] = self.tokenizer.pad_token_id
            labels[np.where(~mask.astype(bool))] = -100
        else:
            # labels[np.where(special_tokens_mask)] = self.tokenizer.pad_token_id
            labels[np.where(special_tokens_mask)] = -100

        # remove mask tokens that are not starts of spans
        to_remove = (mask == 1) & np.roll((mask == 1), 1, 1)
        new_inputs = np.full_like(
            labels, fill_value=self.tokenizer.pad_token_id)

        # splits = list(map(lambda x: x.reshape(-1),  np.split(inputs_copy, indices_or_sections=2, axis=0))
        for i, example in enumerate(np.split(inputs, indices_or_sections=new_inputs.shape[0], axis=0)):
            new_example = example[0][~to_remove[i]]
            new_inputs[i, 0: new_example.shape[0]] = new_example

        # batching now fixed
        return new_inputs, labels
