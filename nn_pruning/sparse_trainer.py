# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Sparse Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

from typing import Dict
from transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from .patch_coordinator import ModelPatchingCoordinator
from collections import defaultdict
import torch.cuda
import torch.nn as nn

from nn_pruning.modules.masked_nn import SageLinearPruningContextModule, MaskedLinear

class TimingModule(nn.Module):
    def __init__(self, model, repeat=1, method_list=None):
        super().__init__()
        self.model = model
        self.reset()
        self.repeat = repeat
        if method_list:
            method_list = list(method_list)
            for method_name in method_list:
                setattr(self, method_name, getattr(model, method_name))

    def reset(self):
        self.elapsed = 0
        self.elapsed_count = 0

    def get_results(self):
        return self.elapsed, self.elapsed_count

    def forward(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for r in range(self.repeat):
            ret = self.model(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()

        self.elapsed += start.elapsed_time(end)
        self.elapsed_count += 1

        return ret

class SparseTrainer:
    def __init__(self, sparse_args):
        self.sparse_args = sparse_args
        self.loss_counter = 0
        self.metrics = defaultdict(float)

    def set_patch_coordinator(self, patch_coordinator: ModelPatchingCoordinator):
        self.patch_coordinator = patch_coordinator

    def log(self, logs: Dict[str, float]) -> None:
        add = {self.log_prefix + k: v for k, v in self.patch_coordinator.log().items()}

        logs.update(add)

        if self.loss_counter != 0:
            for k, v in self.metrics.items():
                if "max" in k or "min" in k:
                    logs[k] = float(v)
                else:
                    # print(k, v, self.loss_counter)
                    logs[k] = float(v) / self.loss_counter


            self.loss_counter = 0
            self.metrics = defaultdict(float)

        return super().log(logs)

    def schedule_threshold(self, training: bool):
        step = self.state.global_step
        self.patch_coordinator.schedule_threshold(step, self.state.max_steps, self.args.warmup_steps, training)

    # def training_step(self, *args, **kwargs):
    #     self.schedule_threshold(True)
    #     self.log_prefix = ""
        # return super().training_step(*args, **kwargs)

        
    def training_step(self, model, inputs):
        self.schedule_threshold(True)
        self.log_prefix = ""
        results = super().training_step( model, inputs)

        # self.update_sage(model)

        # if self.sparse_args.zero_pruned and self.state.global_step > self.args.warmup_steps:
        #     self.zero_pruned_weights(model)

        return results
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        self.metrics["ce_loss"] += float(loss)
        loss, distil_loss = self.patch_coordinator.distil_loss_combine(loss, inputs, outputs)
        self.metrics["distil_loss"] += float(distil_loss)
        regu_loss, lamb, weight_lamb, info = self.patch_coordinator.regularization_loss(model)

        for kind, values in info.items():
            if kind == "total":
                suffix = ""
            else:
                suffix = "_" + kind

            for k, v in values.items():
                self.metrics[k + suffix] += float(v)

        self.loss_counter += 1

        loss = loss + regu_loss * lamb

        return (loss, outputs) if return_outputs else loss

    def evaluate(self, *args, **kwargs):
        # ABSOLUTELY ensure that there are no logs from training to taint these stats
        # will affect stats if epoch doesn't divide into training logs evenly
        self.pre_evaluate()
        self.loss_counter = 0
        self.metrics = defaultdict(float)

        orig_ls = self.model.label_smoothing
        self.model.label_smoothing = 0

        self.schedule_threshold(False)
        # self.log_prefix = "eval_"
        self.log_prefix = "eval/"
        ret = super().evaluate(*args, **kwargs)
        self.model.label_smoothing = orig_ls
        self.post_evaluate()
        return ret

    def pre_evaluate(self):
        pass

    def post_evaluate(self):
        pass

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        assert self.optimizer is None
        self.optimizer = self.create_optimizer(self.model)

        assert self.lr_scheduler is None
        self.lr_scheduler = self.create_scheduler(num_training_steps)

    def create_optimizer(self, model):
        args = self.args

        optimizer_grouped_parameters = self.patch_coordinator.create_optimizer_groups(model, self.args, self.sparse_args)

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        
        return optimizer

    def create_scheduler(self, num_training_steps: int):
        if self.sparse_args.schedule_type not in ["linear", "constant", "cos"]:
            raise RuntimeError("Schedule not recognized")
        
        if self.sparse_args.schedule_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
            )
        elif self.sparse_args.schedule_type == "constant":
            scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
            )
        elif self.sparse_args.schedule_type == "cos":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=5,
            )
        
        return scheduler
