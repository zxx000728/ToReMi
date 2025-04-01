import os
import json
import torch
import math
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path


class PretrainTrainer:
    def __init__(self,
                 args,
                 model,
                 optimizer,
                 lr_scheduler,
                 train_dataloader,
                 eval_dataloader,
                 logger,
                 accelerator,
                 tokenizer,
                 max_grad_norm=0.0,
                 from_checkpoint=None,
                 test_dataloader=None,
                 metadatas=None
                 ):

        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.completed_steps = 0
        self.accelerator = accelerator
        self.tokenizer = tokenizer
        self.max_grad_norm = max_grad_norm
        self._train_iter = iter(train_dataloader)
        self.from_checkpoint = from_checkpoint

        self.metadata_weights = {metadata: 1.0 for metadata in metadatas}
        self.metadata_loss_ema = {metadata: 0.0 for metadata in metadatas}
        self.metadata_loss = {metadata: 0.0 for metadata in metadatas}
        self.metadata_loss_avg = {metadata: 0.0 for metadata in metadatas}
        self.metadata_sample_count = {metadata: 0 for metadata in metadatas}
        self.alpha = 0.9
        self.global_loss_ema = 0.0
        self.global_loss_avg = 0.0
        self.MIN_WEIGHT = 0.1
        self.MAX_WEIGHT = 5.0

        self.training_stage = 0
        self.previous_valid_loss = 0.0
        self.min_valid_decrease_rate = 0.01

    def _save_model(self, save_path=None):
        if save_path is None:
            save_path = self.args.output_dir

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                save_path, save_function=self.accelerator.save)

    def _save_trained(self, save_path=None):
        if save_path is None:
            save_path = self.args.output_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.optimizer.state_dict(),
                   os.path.join(save_path, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(),
                   os.path.join(save_path, "scheduler.pt"))
        trainer_state = {
            "completed_steps": self.completed_steps,
        }
        if self.accelerator.is_main_process:
            with open(os.path.join(save_path, "trainer_state.json"), "w") as f:
                json.dump(trainer_state, f)
        self._save_model(save_path=save_path)

    def calculate_loss_and_accuracy_and_ppl(self, outputs, labels):
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

    def evaluate(self):
        valid_loss_list, valid_acc_list, valid_ppl_list = [], [], []
        self.model.eval()
        with torch.no_grad():
            for eval_step, batch in enumerate(self.eval_dataloader):
                input_batch = {k: v for k,
                               v in batch.items() if k != 'metadata'}
                input_batch = self._move_to_device(input_batch)
                outputs = self.model(**input_batch, return_dict=True)
                labels = input_batch["labels"]
                loss, acc, ppl = self.calculate_loss_and_accuracy_and_ppl(
                    outputs, labels)
                valid_loss_list.append(float(loss))
                valid_acc_list.append(float(acc))
                valid_ppl_list.append(float(ppl))
                self.logger.info("Eval batch {}/{}, loss {}, accuracy {}, ppl {}".format(
                    eval_step + 1, len(self.eval_dataloader), loss, acc, ppl))
        valid_mean_loss = np.mean(valid_loss_list)
        valid_mean_acc = np.mean(valid_acc_list)
        valid_mean_ppl = np.mean(valid_ppl_list)
        self.logger.info("Valid, training step {}/{}, loss {}, accuracy {}, ppl {}".format(
            self.completed_steps, self.args.max_train_steps, valid_mean_loss, valid_mean_acc, valid_mean_ppl
        ))

        # 根据valid loss的变化率调整训练阶段
        if self.previous_valid_loss == 0.0:
            self.previous_valid_loss = valid_mean_loss
        else:
            loss_decrease_rate = (
                self.previous_valid_loss - valid_mean_loss) / self.previous_valid_loss
            # if loss_decrease_rate <= self.min_valid_decrease_rate:
            if self.completed_steps > 7000:
                self.training_stage = 1
                self.logger.info("Change training stage to 1!")
            self.previous_valid_loss = valid_mean_loss

        if self.accelerator.is_main_process:
            wandb.define_metric("valid/*", step_metric='train/step')
            log_dict = {
                "valid/loss": valid_mean_loss,
                "valid/accuracy": valid_mean_acc,
                "valid/ppl": valid_mean_ppl
            }
            wandb.log(log_dict)
        self.accelerator.wait_for_everyone()
        self.model.train()

    def _move_to_device(self, batch):
        device_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                device_batch[k] = v.to(self.args.device)
            else:
                device_batch[k] = v
        return device_batch

    def _get_batch(self):
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_dataloader)
            batch = next(self._train_iter)

        # 取出 collated_batch 和 metadata
        collated_batch = batch["collated_batch"]
        metadata = batch["metadata"]
        collated_batch["metadata"] = metadata
        return self._move_to_device(collated_batch)

    def compute_loss(self):
        self.model.train()
        batch = self._get_batch()
        input_batch = {k: v for k, v in batch.items() if k != 'metadata'}
        outputs = self.model(**input_batch)

        # Calculate per-sample loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_batch['labels'][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_per_sample = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss_per_sample = loss_per_sample.view(
            shift_labels.size(0), -1).sum(dim=1) / shift_labels.size(1)

        if self.args.with_labels:
            # Adjusted loss calculation
            # Create a tensor for all sample weights, initialized with zeros
            sample_weights = torch.zeros(
                len(batch["metadata"]), device=self.accelerator.device)

            # Iterate through metadata and compute the weights
            for i, sample_metadata in enumerate(batch["metadata"]):
                if sample_metadata:  # If metadata is not empty
                    # 样本所有标签的weight取平均作为最终标签weight
                    total_weight = sum(
                        self.metadata_weights[metadata] for metadata in sample_metadata)
                    avg_weight = total_weight / len(sample_metadata)
                    sample_weights[i] = avg_weight

                    # Update metadata_loss and metadata_sample_count
                    for metadata in sample_metadata:
                        # 更新每个标签的loss，包含该标签的所有样本的loss之和
                        self.metadata_loss[metadata] += loss_per_sample[i].item()
                        # 记录包含该标签的样本数量
                        self.metadata_sample_count[metadata] += 1

            # Vectorized weighted loss calculation
            weighted_loss = loss_per_sample * sample_weights
            adjusted_loss = weighted_loss.mean() / self.args.gradient_accumulation_steps

            # Backpropagation
            self.accelerator.backward(adjusted_loss)

            return adjusted_loss.item()
        else:
            loss = loss_per_sample.mean() / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)
            return loss.item()

    def _prepare_from_checkpoint(self):
        if self.from_checkpoint is None:
            return

        state_file = os.path.join(self.from_checkpoint, "trainer_state.json")
        optim_file = os.path.join(self.from_checkpoint, "optimizer.pt")
        sched_file = os.path.join(self.from_checkpoint, "scheduler.pt")
        if os.path.exists(sched_file):
            sched_state = torch.load(sched_file)
            self.lr_scheduler.load_state_dict(sched_state)
        if not os.path.exists(state_file):
            return

        with open(state_file, "r") as f:
            state = json.load(f)
            self.pre_completed_steps = state["completed_steps"]

        self.logger.info(f"Pretrained steps: {self.pre_completed_steps}")
        self.accelerator.wait_for_everyone()

    def compute_metadata_loss(self):
        total_avg_loss = 0.0
        num_valid_metadata = 0  # 用于统计有多少个metadata实际上有样本
        for metadata, loss in self.metadata_loss.items():
            if self.metadata_sample_count[metadata] > 0:
                avg_metadata_loss = loss / \
                    self.metadata_sample_count[metadata]  # 当前周期该标签的平均loss
                total_avg_loss += avg_metadata_loss
                num_valid_metadata += 1
            else:
                avg_metadata_loss = 0.0

            # 更新当前周期每个标签的平均损失
            self.metadata_loss_avg[metadata] = avg_metadata_loss

            # 更新每个标签的损失EMA
            if self.metadata_loss_ema.get(metadata, 0.0) == 0.0:
                self.metadata_loss_ema[metadata] = avg_metadata_loss
            else:
                self.metadata_loss_ema[metadata] = self.alpha * \
                    self.metadata_loss_ema[metadata] + \
                    (1 - self.alpha) * avg_metadata_loss

        # 计算所有标签的平均损失
        if num_valid_metadata > 0:
            avg_loss = total_avg_loss / num_valid_metadata
        else:
            avg_loss = 0.0

        # 更新当前周期所有标签的平均loss
        self.global_loss_avg = avg_loss

        # 更新全局EMA损失为所有标签的平均损失
        if self.global_loss_ema == 0.0:
            self.global_loss_ema = avg_loss
        else:
            self.global_loss_ema = self.alpha * \
                self.global_loss_ema + (1 - self.alpha) * avg_loss

    def adjust_weights(self):
        self.compute_metadata_loss()
        if not self.global_loss_avg:
            self.logger.info("global_loss_avg == 0.0, return")
            return

        for metadata, loss in self.metadata_loss_avg.items():
            # 比较每个标签的损失与全局平均损失，并根据差距调整权重大小
            loss_diff = loss - self.global_loss_avg

            # 训练初期，loss比平均大的加权，loss比平均小的不变
            if self.training_stage == 0:
                if loss_diff > 0:
                    weight_adjustment = 1 + (loss_diff / self.global_loss_avg)
                    self.metadata_weights[metadata] = min(
                        self.metadata_weights[metadata] * weight_adjustment, self.MAX_WEIGHT)
            # 训练后期，loss比平均大的降权，loss比平均小的加权
            else:
                # 如果loss比平均值大，降权
                if loss_diff > 0:
                    weight_adjustment = 1 - (loss_diff / self.global_loss_avg)
                    self.metadata_weights[metadata] = max(
                        self.metadata_weights[metadata] * weight_adjustment, self.MIN_WEIGHT)
                # 如果loss比平均值小，加权
                else:
                    weight_adjustment = 1 + \
                        abs(loss_diff / self.global_loss_avg)
                    self.metadata_weights[metadata] = min(
                        self.metadata_weights[metadata] * weight_adjustment, self.MAX_WEIGHT)

            self.logger.info(
                "Training step {}, Metadata {}, sample_count {}, metadata_loss_avg {}, global_loss_avg {}, new_weight {}, metadata_loss_ema {}, global_loss_ema {}".format(
                    self.completed_steps,
                    metadata,
                    self.metadata_sample_count[metadata],
                    loss,
                    self.global_loss_avg,
                    self.metadata_weights[metadata],
                    self.metadata_loss_ema[metadata],
                    self.global_loss_ema
                )
            )

        # 重置累积损失
        self.metadata_loss = {
            metadata: 0.0 for metadata in self.metadata_loss.keys()}
        self.metadata_sample_count = {
            metadata: 0 for metadata in self.metadata_loss.keys()}
        self.metadata_loss_avg = {
            metadata: 0.0 for metadata in self.metadata_loss.keys()}
        self.global_loss_avg = 0.0

    def update(self, tr_loss, loss_step):
        if self.completed_steps % self.args.steps_to_log == 0:
            self.logger.info(
                "Training step {}, learning rate {}, average loss {}".format(
                    self.completed_steps,
                    self.optimizer.param_groups[0]["lr"],
                    tr_loss / loss_step
                )
            )
            if self.accelerator.is_main_process:
                wandb.define_metric("train/*", step_metric='train/step')
                log_dict = {
                    "train/step": self.completed_steps,
                    "train/loss": tr_loss / loss_step,
                    "train/lr": self.optimizer.param_groups[0]["lr"]
                }
                wandb.log(log_dict)
            tr_loss = 0.0
            loss_step = 0

        if self.completed_steps % self.args.steps_to_eval == 0:
            self.evaluate()

        if self.completed_steps % self.args.steps_to_save == 0:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                self._save_trained(
                    save_path=os.path.join(
                        self.args.output_dir, 'checkpoint-{}'.format(self.completed_steps))
                )

        # 调整权重
        if self.completed_steps % self.args.steps_to_adjust_weight == 0 and self.completed_steps and self.args.with_labels:
            self.adjust_weights()

    def train(self):
        total_batch_size = self.args.per_device_train_batch_size * \
            self.args.gradient_accumulation_steps * self.accelerator.num_processes
        self.logger.info("***** Running training *****")
        self.logger.info(
            f"  Num examples = {len(self.train_dataloader.dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        self.logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(
            f"  Total optimization steps = {self.args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps))
        self.completed_steps = 0
        self.pre_completed_steps = 0
        self._prepare_from_checkpoint()
        tr_loss = 0.0
        loss_step = 0
        for step in range(self.args.max_train_steps * self.args.gradient_accumulation_steps):
            if self.completed_steps < self.pre_completed_steps:
                self._get_batch()
                if step % self.args.gradient_accumulation_steps == 0:
                    self.completed_steps += 1
                    progress_bar.update(1)
                continue
            if step % self.args.gradient_accumulation_steps == 0:
                tr_loss += self.compute_loss()
                loss_step += 1
                if self.max_grad_norm > 0:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                self.completed_steps += 1
                self.update(tr_loss, loss_step)
                tr_loss = 0.0
                loss_step = 0
            else:
                with self.accelerator.no_sync(self.model):
                    tr_loss += self.compute_loss()
                    loss_step += 1

        self.evaluate()
        self._save_trained(
            save_path=os.path.join(self.args.output_dir, 'final')
        )
