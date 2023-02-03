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
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nn_functional
from transformers import AutoConfig, AutoModelForQuestionAnswering
from dataclasses import dataclass, field
from collections import defaultdict

from nn_pruning.model_structure import struct_from_config, ModelStructureNotFound

from .modules.masked_nn import (
    ChannelPruningModulePatcher,
    JointPruningModulePatcher,
    LinearPruningModulePatcher,
    LinearPruningArgs,
    MaskedLinear,
    MaskedLinearModelCompiler,
    GenericLinearPruningContextModule,
    head_mask,
    InitDirective
)
from .modules.nonorm import Layer2NoNorm, NoNorm, NoNormCompiler, Layer2NoNormPatcher
from .modules.gelu2relu import GeLU2ReLUModelPatcher
from .inference_model_patcher import BertHeadsPruner

from nn_pruning.training_patcher import (
    LinearModelPatcher,
    PatcherContext,
    PatcherContextModule,
)
from pathlib import Path
import json


@dataclass
class SparseTrainingArguments:
    """
    Sparse training specific arguments
    """

    soft_temperature: float = field(
        default=1e-3, metadata={"help": "soft_temperature."}
    )

    span_reg_lamb: float = field(
        default=0, metadata={"help": "span_reg_lamb. 0 means no span reg."}
    )

    running_r2_mult: float = field(
        default=1, metadata={"help": "running_r2_mult."}
    )

    A_reg_lamb: float = field(
        default=0, metadata={"help": "A_reg_lamb. 0 means no A reg."}
    )

    span_reg_A_learning_rate: float = field(
        default=1e-2, metadata={"help": "The initial learning rate for span_reg_A."}
    )

    adjust_grad_lamb: float = field(
        default=0, metadata={"help": "adjust grad lamb."}
    )

    anti_gum: bool = field(
        default=False,
        metadata={"help": "use anti-gum"},
    )

    cpu_cos_sim: bool = field(
        default=False,
        metadata={"help": "cos sim stored on cpu"},
    )
    
    mask_frozen: bool = field(
        default=False,
        metadata={"help": "mask_frozen and not trained"},
    )

    mask_span_reg_lamb: float = field(
        default=0, metadata={"help": "adjust mask reg with span reg."}
    )

    opt_span_reg_only: bool = field(
        default=False,
        metadata={"help": "optimize only the As for span reg"},
    )

    adjust_grad_do_mult: bool = field(
        default=False,
        metadata={"help": "adjust_grad_do_mult"},
    )
    
    adjust_mask_grad: bool = field(
        default=False,
        metadata={"help": "adjust_mask_grad"},
    )

    uniqueness_reg_mask: bool = field(
        default=False,
        metadata={"help": "uniqueness_reg_mask"},
    )

    running_cos_mult: float = field(
        default=1, metadata={"help": "running_cos_mult."}
    )

    running_cos_method: float = field(
        default=1, metadata={"help": "running_cos_method."}
    )

    track_eval_cos: bool = field(
        default=False,
        metadata={"help": "track_eval_cos"},
    )

    scale_pruned: bool = field(
        default=False,
        metadata={"help": "scale_pruned"},
    )

    scale_params_learning_rate: float = field(
        default=1e-2, metadata={"help": "The learning rate for scaling params."}
    )

    scale_fc: bool = field(
        default=False,
        metadata={"help": "scale_fc"},
    )

    scale_proj: bool = field(
        default=False,
        metadata={"help": "scale_proj"},
    )

    train_only_bias_ln:bool = field(
        default=False,
        metadata={"help": "train only bias, ln params"},
    )

    mask_scores_learning_rate: float = field(
        default=1e-2, metadata={"help": "The initial learning rate for mask_scores."}
    )

    sage_beta_meta: float = field(
        default=.1, metadata={"help": "Sage Beta Meta."}
    )

    sage_beta_3: float = field(
        default=.1, metadata={"help": "Sage Beta 3."}
    )

    sage_delta_T: int = field(
        default=10, metadata={"help": "Sage Delta T."}
    )

    zero_pruned: bool = field(
        default=False,
        metadata={"help": "zero out pruned weights after warmup ends"},
    )

    dense_pruning_method: str = field(default="topK", metadata={"help": "Dense Layers pruning method."})

    attention_pruning_method: str = field(default="topK", metadata={"help": "Dense Layers pruning method."})

    ampere_pruning_method: str = field(
        default="disabled",
        metadata={"help": "Ampere sparse method ('disabled' for no ampere sparsity, topK, annealing, sigmoied_threshold, threshold)"},
    )

    attention_output_with_dense: bool = field(
        default=True,
        metadata={"help": "share the same pruning parameters for attention output and other dense matrices"},
    )

    bias_mask: bool = field(
        default=True,
        metadata={"help": "Apply the mask built on weight to the bias too (not doing so will impact somewhat ability to prune complete heads, as bias is then pruned while being nonzero)"},
    )

    mask_init: str = field(default="constant", metadata={"help": "Mask scores initialization method"})

    mask_scale: float = field(
        default=0.0,
        metadata={"help": "Parameter to use with mask_init."},
    )

    dense_block_rows: int = field(
        default=1,
        metadata={"help": "Block size in rows for dense layers."},
    )

    dense_block_cols: int = field(
        default=1,
        metadata={"help": "Block size in cols for dense layers."},
    )

    attention_block_rows: int = field(
        default=1,
        metadata={"help": "Block size in rows for attention."},
    )

    attention_block_cols: int = field(
        default=1,
        metadata={"help": "Block size in cols for attention."},
    )

    initial_threshold: float = field(
        default=1.0,
        metadata={"help": "Initial value of the threshold (for scheduling)."},
    )

    final_threshold: float = field(
        default=0.5,
        metadata={"help": "Final value of the threshold (for scheduling)."},
    )

    initial_warmup: float = field(
        default=1,
        metadata={
            "help": "Run `initial_warmup` * `warmup_steps` steps of threshold warmup during which threshold stays at its `initial_threshold` value (sparsity schedule)."
        },
    )
    final_warmup: float = field(
        default=2,
        metadata={
            "help": "Run `final_warmup` * `warmup_steps` steps of threshold cool-down during which threshold stays"
        },
    )

    initial_ampere_temperature: float = field(
        default=0.0,
        metadata={"help": "Initial value of the ampere temperature (for scheduling)."},
    )
    final_ampere_temperature: float = field(
        default=20.0,
        metadata={"help": "Final value of the ampere temperature (for scheduling)."},
    )

    weight_regularization: str = field(
        default="disabled",
        metadata={"help": "Add regularization to the weight scores."},
    )

    weight_regularization_final_lambda: float = field(
        default=0.0,
        metadata={"help": "Regularization intensity (used in conjunction with `weight_regularization`)."},
    )

    regularization: str = field(
        default="disabled",
        metadata={"help": "Add L0 or L1 regularization to the mask scores."},
    )

    regularization_final_lambda: float = field(
        default=0.0,
        metadata={"help": "Regularization intensity (used in conjunction with `regularization`)."},
    )

    attention_lambda: float = field(
        default=1.0,
        metadata={"help": "Regularization intensity for attention (attention lambda will be regularization_lambda * attention_lambda)."},
    )

    dense_lambda: float = field(
        default=1.0,
        metadata={
            "help": "Regularization intensity for dense (attention lambda will be regularization_lambda * dense_lambda)."},
    )

    decoder_attention_lambda: float = field(
        default=None,
        metadata={"help": "Regularization intensity for decoder attention (attention lambda will be regularization_lambda * decoder_attention_lambda)."},
    )

    decoder_dense_lambda: float = field(
        default=None,
        metadata={
            "help": "Regularization intensity for decoder dense (attention lambda will be regularization_lambda * decoder_dense_lambda)."},
    )

    distil_teacher_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to the already SQuAD fine-tuned teacher model. Only for distillation."},
    )

    distil_alpha_ce: float = field(
        default=0.5,
        metadata={"help": "Cross entropy loss linear weight. Only for distillation."},
    )

    distil_alpha_teacher: float = field(
        default=0.5,
        metadata={"help": "Distillation loss linear weight. Only for distillation."},
    )

    distil_temperature: float = field(
        default=2.0,
        metadata={"help": "Distillation temperature. Only for distillation."},
    )

    final_finetune: bool = field(
        default=False,
        metadata={
            "help": "Run a final fine tune pass on the network"
        },
    )

    layer_norm_patch: bool = field(
        default=False,
        metadata={
            "help": "Transform the LayerNorms in a MobileBert NoNorm"
        },
    )
    layer_norm_patch_steps: int = field(
        default=50000,
        metadata={
            "help": "Number of steps for the transition from LayerNorm to NoNorm"
        },
    )

    layer_norm_patch_start_delta: int = field(
        default=0.99,
        metadata={
            "help": "Starting smoothing factor for transition from LayerNorm to NoNorm (final is 1.0)."
        },
    )

    gelu_patch: bool = field(
        default=False,
        metadata={
            "help": "Transform the GeLU in ReLU"
        },
    )

    gelu_patch_steps: int = field(
        default=50000,
        metadata={
            "help": "Number of steps for the transition from GeLU to ReLU"
        },
    )

    linear_min_parameters: int = field(
        default=0.005,
        metadata={
            "help": "Minimum fraction of parameters which should be non zero when using ThresholdBinarizer"
        },
    )

    rewind_model_name_or_path: str = field(
        default=None,
        metadata={
           "help": "Model that will be used as a guide to prevent pruning of some attention heads while redoing fine-pruning."
        },
    )

    eval_with_current_patch_params: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep the transition parameters used during training for eval. Only for Layer2NoNorm, GeLU2ReLU and pruning threshold."
        },
    )

    qat: bool = field(
        default=False,
        metadata={"help": "Whether to prepare the model for Quantization Aware Training"},
    )

    qconfig: str = field(
        default="default",
        metadata={"help": "The quantization scheme configuration to use for QAT"},
    )

    schedule_type: str = field(default="linear", metadata={"help": "LR Schedule type."})

    @classmethod
    def hybrid(cls, regularization_lambda):
        sparse_args = cls()
        sparse_args.dense_pruning_method = "sigmoied_threshold:1d_alt"
        sparse_args.attention_pruning_method = "sigmoied_threshold"
        sparse_args.attention_block_rows = 32
        sparse_args.attention_block_cols = 32
        sparse_args.attention_output_with_dense = False
        sparse_args.initial_threshold = 0.0
        sparse_args.final_threshold = 0.1

        sparse_args.regularization = "l1"
        sparse_args.regularization_final_lambda = regularization_lambda
        return sparse_args


class ModelPatchingCoordinator:

    def __init__(self, sparse_args, device, cache_dir, model_name_or_path, logit_names, teacher_constructor):
        # logit_names is ["start_logits", "end_logits"] for qa, ["logits"] for glue etc
        # teacher model is AutoModelForQuestionAnswering for qa, AutoModelForSequenceClassification for glue etc
        self.sparse_args = sparse_args
        self.patcher_context = PatcherContext()
        self.teacher_constructor = teacher_constructor
        self.device = device
        self.cache_dir = cache_dir
        self.teacher = None
        self.layer_head_mask = self.create_head_rewind_info(device, cache_dir)
        self.logit_names = logit_names
        self.model_name_or_path = model_name_or_path
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model_structure = struct_from_config(config.__class__)

    def parse_pruning_method(self, method):
        parts = method.split(":")
        if len(parts) == 2:
            return parts
        elif len(parts) == 1:
            return parts[0], "default"
        else:
            raise RuntimeError("Could not parse pruning method")

    def log(self):
        logs = {}
        for k, v in self.patcher_context.enumerate_context_data():
            logs[k] = v

        return logs

    def create_teacher(self):
        if self.teacher is not None:
            return self.teacher

        device = self.device
        cache_dir = self.cache_dir

        sparse_args = self.sparse_args

        if sparse_args.distil_teacher_name_or_path is not None:
            assert sparse_args.distil_alpha_ce > 0.0
            assert sparse_args.distil_alpha_ce + sparse_args.distil_alpha_teacher > 0.0

            model_config = AutoConfig.from_pretrained(sparse_args.distil_teacher_name_or_path, cache_dir=cache_dir)

            teacher = self.teacher_constructor.from_pretrained(
                sparse_args.distil_teacher_name_or_path,
                from_tf=bool(".ckpt" in sparse_args.distil_teacher_name_or_path),
                config=model_config,
                cache_dir=cache_dir,
            )
            teacher.to(device)
            self.teacher = teacher

            for n, p in teacher.named_parameters():
                p.requires_grad = False

        return self.teacher


    def create_head_rewind_info(self, device, cache_dir):
        if not hasattr(self.sparse_args, "rewind_model_name_or_path"):
            return None

        rewind_model_name_or_path = self.sparse_args.rewind_model_name_or_path
        if rewind_model_name_or_path is None:
            return None
        else:
            rewind_config = AutoConfig.from_pretrained(rewind_model_name_or_path, cache_dir=cache_dir)

            return head_mask(rewind_config, device)

    def schedule_threshold(
        self,
        step: int = -1,
        total_step: int = -1,
        warmup_steps: int = -1,
        training: bool = False,
        compile:bool = False,
    ):
        sparse_args = self.sparse_args

        initial_threshold = sparse_args.initial_threshold
        final_threshold = sparse_args.final_threshold
        initial_warmup = sparse_args.initial_warmup
        final_warmup = sparse_args.final_warmup
        final_lambda = sparse_args.regularization_final_lambda
        final_weight_lambda = sparse_args.weight_regularization_final_lambda
        initial_ampere_temperature = sparse_args.initial_ampere_temperature
        final_ampere_temperature = sparse_args.final_ampere_temperature


        if not training:
            step -= 1

        eval_with_current_patch_params = (hasattr(sparse_args, "eval_with_current_patch_params") and sparse_args.eval_with_current_patch_params)
        use_scheduler = training or eval_with_current_patch_params

        if compile:
            if use_scheduler:
                base_path = Path(self.model_name_or_path)
                training_args = torch.load(str(base_path / "training_args.bin"))
                warmup_steps = training_args.warmup_steps

                with (base_path / "trainer_state.json").open() as f:
                    trainer_state = json.load(f)

                step = trainer_state["global_step"]
                total_step = trainer_state["max_steps"]

        if use_scheduler:
            if step <= initial_warmup * warmup_steps:
                mul_coeff = 1.0
                threshold = initial_threshold
                ampere_temperature = initial_ampere_temperature
            elif step > (total_step - final_warmup * warmup_steps):
                mul_coeff = 0.0
                threshold = final_threshold
                ampere_temperature = final_ampere_temperature
            else:
                spars_warmup_steps = initial_warmup * warmup_steps
                spars_schedu_steps = (final_warmup + initial_warmup) * warmup_steps
                mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
                threshold = final_threshold + (initial_threshold - final_threshold) * (mul_coeff ** 3)
                ampere_temperature = final_ampere_temperature + (
                    initial_ampere_temperature - final_ampere_temperature
                ) * (mul_coeff ** 3)
        else:
            mul_coeff = 0.0
            threshold = final_threshold
            ampere_temperature = final_ampere_temperature
        
        # if sparse_args.dense_pruning_method == "topK":
        if False:
        # if True:
            # regu_lambda = final_lambda *  (1-threshold)
            # weight_regu_lambda = final_weight_lambda *  (1-threshold)
            0/0
        else:
            regu_lambda = final_lambda * threshold / final_threshold
            weight_regu_lambda = final_weight_lambda * threshold / final_threshold
            # 0/0
            # 0/0
            # 0/0

        context_data = dict(
            threshold=threshold,
            regu_lambda=regu_lambda,
            weight_regu_lambda=weight_regu_lambda,
            ampere_temperature = ampere_temperature,
            progress = 1.0 - mul_coeff
        )


        def interp(a,b, interpf):
            return a * interpf + (1.0 - interpf) * b

        if hasattr(sparse_args, "layer_norm_patch") and sparse_args.layer_norm_patch:
            if use_scheduler:
                interpf = 0.0
                layer_norm_patch_steps = sparse_args.layer_norm_patch_steps
                if step < layer_norm_patch_steps:
                    interpf = 1.0 - (step / layer_norm_patch_steps)

                delta = interp(sparse_args.layer_norm_patch_start_delta, 1.0, interpf)
                mix = interpf

                context_data["layernorm_to_nonorm_delta"] = delta
                context_data["layernorm_to_nonorm_mix"] = mix
            else:
                context_data["layernorm_to_nonorm_delta"] = 1.0
                context_data["layernorm_to_nonorm_mix"] = 0.0

        if hasattr(sparse_args, "gelu_patch") and sparse_args.gelu_patch:
            if use_scheduler:
                interpf = 0.0
                gelu_patch_steps = sparse_args.gelu_patch_steps
                if step < gelu_patch_steps:
                    interpf = 1.0 - (step / gelu_patch_steps)

                context_data["gelu_to_relu_mix"] = interpf
            else:
                context_data["gelu_to_relu_mix"] = 0.0

        self.patcher_context.set_context_data_dict(context_data)

    def regularization_loss(self, model: nn.Module):
        # Return regularization, lambda, and information on the network sparsity
        mode = self.sparse_args.regularization

        info = {}

        regul_modes = ["l1", "l0"]
        # if mode in regul_modes:
        #     threshold = self.patcher_context.get_context_data("threshold")

        for name, module in model.named_modules():
            module_regu = 0
            module_nnz_info = {"nnz":0, "numel":0}
            nummod = 1
            # if isinstance(module, nn.Linear): print("found")
            if mode not in regul_modes:
                # if isinstance(module, nn.Linear):
                #     weight = module.weight
                #     module_nnz_info["nnz"] = (weight != 0).sum()
                #     module_nnz_info["numel"] = weight.numel()
                # elif isinstance(module, MaskedLinear): # added - why wasn't this here before? sparsity only with regu?
                #     module_nnz_info = module.get_sparsity_info()
                #     nummod = 1
                if isinstance(module, MaskedLinear): # added - why wasn't this here before? sparsity only with regu?
                    if module.args.method != "disabled": 
                        module_nnz_info = module.get_sparsity_info()
                        nummod = 1
                else:
                    continue
            elif isinstance(module, GenericLinearPruningContextModule):
                module_regu = module.regularization(mode)
            elif isinstance(module, MaskedLinear):
                if module.args.method != "disabled":
                    module_nnz_info = module.get_sparsity_info()
                    nummod = 1
            elif hasattr(module, "regularization"):
                module_regu = module.regularization()
                if hasattr(module, "get_sparsity_info"):
                    module_nnz_info = module.get_sparsity_info()
            else:
                continue

            key = "decoder_" if self.model_structure.is_decoder(name) else ""
            exclude_att_dense = not hasattr(self.sparse_args, "attention_output_with_dense") or self.sparse_args.attention_output_with_dense
            key += "attention" if self.model_structure.is_attention(name, exclude_att_dense=exclude_att_dense) else "dense"

            # print(name, key, module_nnz_info) # debug

            if key not in info:
                info[key] = defaultdict(float)

            key_info = info[key]
            key_info["regu"] += module_regu
            key_info["nummod"] += nummod

            for k,v in module_nnz_info.items():
                key_info[k] += float(v)

        # print(info) # debug
        # print(info["attention"]) # debug

        if mode not in regul_modes:
            lamb = 0
            weight_lamb = self.patcher_context.get_context_data("weight_regu_lambda")
            lambdas = {k: 0 for k in info.keys()}
        else:
            lamb = self.patcher_context.get_context_data("regu_lambda")
            weight_lamb = self.patcher_context.get_context_data("weight_regu_lambda")
            lambdas = {}
            n = len(info)
            for k in info.keys():
                if k.endswith('attention'):
                    if k.startswith('decoder'):
                        if self.sparse_args.decoder_attention_lambda is None:
                            self.sparse_args.decoder_attention_lambda = self.sparse_args.attention_lambda
                        lambdas[k] = self.sparse_args.decoder_attention_lambda / n
                    else:
                        lambdas[k] = self.sparse_args.attention_lambda / n
                else:
                    if k.startswith('decoder'):
                        if self.sparse_args.decoder_dense_lambda is None:
                            self.sparse_args.decoder_dense_lambda = self.sparse_args.dense_lambda
                        lambdas[k] = self.sparse_args.decoder_dense_lambda / n
                    else:
                        lambdas[k] = self.sparse_args.dense_lambda / n

        info["total"] = defaultdict(float)

        for key, value in info.items():
            if key == "total":
                continue
            for k, v in value.items():
                if k == "numel" or "nnz" in k:
                    info["total"][k] += v

        for key, value in info.items():
            if value["numel"] != 0:
                # No patching (no pruning) -> no information on nnz -> dense linear layers
                value["nnz_perc"] = value["nnz"] / value["numel"]
            else:
                value["nnz_perc"] = 1.0
            for k in "nnz", "numel":
                if k in value:
                    del value[k]
            if key == "total":
                continue
            if value["nummod"] != 0:
                value["regu_loss"] = value["regu"] * lambdas[key] / value["nummod"]
                info["total"]["regu_loss"] += value["regu_loss"]
            for k in "regu", "nummod":
                if k in value:
                    del value[k]

        return info["total"]["regu_loss"], lamb, weight_lamb, info

    def distil_loss_combine(self, ce_loss, model_inputs, model_outputs):
        sparse_args = self.sparse_args

        if "distil_topk_probs" in model_inputs:
            return self.distil_loss_combine_topk(ce_loss, model_inputs, model_outputs)


        teacher = self.create_teacher()

        if teacher == None:
            return ce_loss, 0.0

        temperature = sparse_args.distil_temperature

        # teacher_inputs_ = model_inputs.copy()
        # if 'labels' in teacher_inputs_:
        #     del teacher_inputs_['labels']

        # teacher_inputs = {}
        # for k,v in teacher_inputs_.items():
        #     teacher_inputs[k] = v.detach().clone()
        
        # print(5, "torch alloc", torch.cuda.memory_allocated(0), "torch alloc 1", torch.cuda.memory_allocated(1), "torch reserved", torch.cuda.memory_reserved(0))


        teacher_inputs = {}
        for k,v in model_inputs.items():
            teacher_inputs[k] = v.detach().clone()
        if 'labels' in teacher_inputs:
            del teacher_inputs['labels']

        with torch.no_grad():
            teacher_outputs = teacher(**teacher_inputs)
            
        # print(6, "torch alloc", torch.cuda.memory_allocated(0), "torch alloc 1", torch.cuda.memory_allocated(1), "torch reserved", torch.cuda.memory_reserved(0))


        # if True:
        #     topk_probs, topk_indices = torch.topk(teacher_outputs["logits"].detach(), k=20, dim=-1)
        #     return self.distil_loss_combine_topk(ce_loss, model_inputs, model_outputs, probs=topk_probs, indices=topk_indices)

        loss_logits = 0
        for logit_name in self.logit_names:
            logits_stu = model_outputs[logit_name]
            # logits_tea = teacher_outputs[logit_name].detach().clone()
            logits_tea = teacher_outputs[logit_name].detach()
            # del teacher_outputs
            # del teacher_inputs
            # del teacher
            # torch.cuda.empty_cache()
            
            # print(7, "torch alloc", torch.cuda.memory_allocated(0), "torch alloc 1", torch.cuda.memory_allocated(1), "torch reserved", torch.cuda.memory_reserved(0))



            loss_logits_part = nn_functional.kl_div(
                # input=nn_functional.log_softmax(logits_stu / temperature, dim=-1),
                # target=nn_functional.softmax(logits_tea / temperature, dim=-1),
                # input=nn_functional.log_softmax(logits_stu.cpu() / temperature, dim=-1),
                # target=nn_functional.softmax(logits_tea.cpu() / temperature, dim=-1),
                input=nn_functional.log_softmax(logits_stu.to("cuda:1") / temperature, dim=-1),
                target=nn_functional.softmax(logits_tea.to("cuda:1") / temperature, dim=-1),
                reduction="batchmean",
            ) * (temperature ** 2)

            loss_logits = loss_logits + loss_logits_part

        loss_logits = loss_logits / len(self.logit_names)

        loss = sparse_args.distil_alpha_teacher * loss_logits.to(ce_loss.device) + sparse_args.distil_alpha_ce * ce_loss

        return loss, loss_logits

    def distil_loss_combine_topk(self, ce_loss, model_inputs, model_outputs, probs=None, indices=None):
        sparse_args = self.sparse_args
        if probs is None:
            teacher_probs = model_inputs["distil_topk_probs"]
        else:
            teacher_probs = probs
        if indices is None:
            teacher_indices = model_inputs["distil_topk_indices"]
        else:
            teacher_indices = indices
        
        student_logits = model_outputs["logits"]

        n_vocab = student_logits.shape[-1]
        topk = teacher_indices.shape[-1]
        missing_probs = (1.0 - teacher_probs.sum(dim=-1))   / (n_vocab - topk)

        teacher_full_probs = torch.ones_like(student_logits) * missing_probs.unsqueeze(-1)
        teacher_full_probs = teacher_full_probs.scatter(-1, teacher_indices, teacher_probs)


        temperature = sparse_args.distil_temperature
        if temperature != 1.0:
            print("can only use temperature 1 for now")
            0/0
        loss_logits = 0
        for logit_name in self.logit_names:  # temp code - will cause issues for datasets with multiple logit names
            logits_stu = model_outputs[logit_name]
            # logits_tea = teacher_full_probs

            loss_logits_part = nn_functional.kl_div(
                input=nn_functional.log_softmax(logits_stu / temperature, dim=-1),
                target=teacher_full_probs,
                reduction="batchmean",
            ) * (temperature ** 2)

            loss_logits = loss_logits + loss_logits_part

        loss_logits = loss_logits / len(self.logit_names)

        loss = sparse_args.distil_alpha_teacher * loss_logits + sparse_args.distil_alpha_ce * ce_loss

        return loss, loss_logits




    def create_optimizer_groups(self, model, args, sparse_args, span_reg_params=[]):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight", "NoNorm.weight", "layer_norm.weight", "layernorm_embedding.weight",
                    # "final_layer_norm.weight", "ln_1.weight", "ln_2.weight", "ln_f.weight"]
                    "final_layer_norm.weight", "ln_1.weight", "ln_2.weight", "ln_f.weight", "span_reg_A", "span_reg_B"]
        mask_params = []
        no_decay_params = []
        decay_params = []
        scale_params = []

        opt_only_sparse_reg = False
        if sparse_args.opt_span_reg_only:
            print("optimizing only span reg")
            opt_only_sparse_reg = True

        for n, p in model.named_parameters():
            if not p.requires_grad:
                print("skip no grad", n)
                continue
            
            no_decay_param = any(nd in n for nd in no_decay)
            if sparse_args.train_only_bias_ln and not no_decay_param:
                p.requires_grad = False
                continue
            
            if "out_scale" in n:
                print("out_scale param add", n, "lr", sparse_args.scale_params_learning_rate)
                scale_params.append(p)
                continue

            if "span_reg" in n:
                print("span reg param add", n)
                span_reg_params.append(p)
                continue

            if "mask_score" in n:
                if sparse_args.mask_frozen:
                    print("mask frozen, not optimizing mask")
                    continue
                if "sage" not in sparse_args.dense_pruning_method: # calculate manually for sage
                    mask_params.append(p)
            elif no_decay_param:
                no_decay_params.append(p)
            elif "span_reg" in n:
                # span_reg_params.append(p)
                0/0 # temp - moving this out of model!
            else:
                decay_params.append(p)
            print("optimizing", n)

        optimizer_grouped_parameters = [
            {
                "params": mask_params,
                "lr": sparse_args.mask_scores_learning_rate,
            },
            {
                "params": no_decay_params,
                "lr": args.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": span_reg_params,
                "lr": sparse_args.span_reg_A_learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": scale_params,
                "lr": sparse_args.scale_params_learning_rate,
                "weight_decay": args.weight_decay,
            },
            {
                "params": decay_params,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            },
        ]

        return optimizer_grouped_parameters


    def patch_model(self, model, trial = None):
        layers_count = model.config.num_hidden_layers
        sparse_args = self.sparse_args

        device = model.device

        attention_pruning_method_parts = self.parse_pruning_method(sparse_args.attention_pruning_method)

        if hasattr(sparse_args, "bias_mask"):
            bias_mask = sparse_args.bias_mask
        else:
            bias_mask = False

        if hasattr(sparse_args, "linear_min_parameters"):
            linear_min_parameters = sparse_args.linear_min_parameters
        else:
            linear_min_parameters = 0.005

        patcher_context = self.patcher_context

        if attention_pruning_method_parts[0] != "disabled" or sparse_args.ampere_pruning_method != "disabled":
        # if True: # test no pruning mask
            args_attention = LinearPruningArgs(
                method=attention_pruning_method_parts[0],
                submethod=attention_pruning_method_parts[1],
                ampere_method=sparse_args.ampere_pruning_method,
                block_rows=sparse_args.attention_block_rows,
                block_cols=sparse_args.attention_block_cols,
                bias_mask=bias_mask,
                min_elements=linear_min_parameters,
                mask_frozen=sparse_args.mask_frozen,
            )

            args_attention_t = LinearPruningArgs(
                method=attention_pruning_method_parts[0],
                submethod=attention_pruning_method_parts[1],
                ampere_method=sparse_args.ampere_pruning_method,
                block_rows=sparse_args.attention_block_cols,
                block_cols=sparse_args.attention_block_rows,
                bias_mask=bias_mask,
                min_elements=linear_min_parameters,
                mask_frozen=sparse_args.mask_frozen,
            )

            if args_attention.submethod == "joint":
                p_attention = JointPruningModulePatcher(patcher_context, args_attention, model_structure=self.model_structure, suffix=".attention")
                p_attention_t = JointPruningModulePatcher(patcher_context, args_attention_t, model_structure=self.model_structure, suffix=".attention")
            else:
                p_attention = LinearPruningModulePatcher(patcher_context,
                                                         args_attention,
                                                         model_structure=self.model_structure,
                                                         row_additive_mask = self.layer_head_mask)
                p_attention_t = LinearPruningModulePatcher(patcher_context,
                                                           args_attention_t,
                                                           model_structure = self.model_structure,
                                                           col_additive_mask = self.layer_head_mask)
        else:
            p_attention = None
            p_attention_t = None

        dense_pruning_method_parts = self.parse_pruning_method(sparse_args.dense_pruning_method)

        if dense_pruning_method_parts[0] != "disabled" or sparse_args.ampere_pruning_method != "disabled":
            args_dense = LinearPruningArgs(
                method=dense_pruning_method_parts[0],
                submethod=dense_pruning_method_parts[1],
                ampere_method=sparse_args.ampere_pruning_method,
                block_rows=sparse_args.dense_block_rows,
                block_cols=sparse_args.dense_block_cols,
                bias_mask=bias_mask,
                min_elements=linear_min_parameters,
                sage_beta_meta=sparse_args.sage_beta_meta,
                soft_temperature=sparse_args.soft_temperature,
                adjust_grad_lamb=sparse_args.adjust_grad_lamb,
                mask_span_reg_lamb=sparse_args.mask_span_reg_lamb,
                scale_pruned=sparse_args.scale_pruned,
                scale_fc=sparse_args.scale_fc,
                scale_proj=sparse_args.scale_proj,
                mask_init=InitDirective(kind=sparse_args.mask_init, scale=sparse_args.mask_scale),
                mask_frozen=sparse_args.mask_frozen,
            )
            if args_dense.submethod.startswith("1d"):
                p_dense = ChannelPruningModulePatcher(
                    patcher_context, args_dense, model_structure=self.model_structure, suffix="dense"
                )
            else:
                p_dense = LinearPruningModulePatcher(patcher_context, args_dense, model_structure=self.model_structure)
        else:
            p_dense = None

        if not hasattr(sparse_args, "attention_output_with_dense") or sparse_args.attention_output_with_dense:
            p_att_dense = p_dense
        else:
            p_att_dense = p_attention_t

        module_patchers = dict(
            query=p_attention,
            key=p_attention,
            value=p_attention,
            att_dense=p_att_dense,
            encoder_decoder_query=p_attention,
            encoder_decoder_key=p_attention,
            encoder_decoder_value=p_attention,
            encoder_decoder_att_dense=p_att_dense,
            interm_dense=p_dense,
            output_dense=p_dense,
        )

        if hasattr(sparse_args, "layer_norm_patch"):
            layer_norm_patch = sparse_args.layer_norm_patch
        else:
            layer_norm_patch = False

        if hasattr(sparse_args, "gelu_patch"):
            gelu_patch = sparse_args.gelu_patch
        else:
            gelu_patch = False

        patcher = LinearModelPatcher(module_patchers, model_structure=self.model_structure)

        patcher.patch(model)
        model = model.to(device)  # TODO: change this by making sure the mask_scores are located at the right place.

        self.stats = {}
        self.stats["main"] = patcher.stats

        if layer_norm_patch:
            def schedule_callback():
                mix = self.patcher_context.get_context_data("layernorm_to_nonorm_mix")
                delta = self.patcher_context.get_context_data("layernorm_to_nonorm_delta")
                return dict(mix=mix, delta=delta)

            layer_norm_patcher = Layer2NoNormPatcher(schedule_callback=schedule_callback)
            layer_norm_patcher.patch(model)
            self.stats["layer_norm"] = layer_norm_patcher.stats

        if gelu_patch:
            def schedule_callback():
                mix = self.patcher_context.get_context_data("gelu_to_relu_mix")
                return dict(mix=mix)

            gelu_patcher = GeLU2ReLUModelPatcher(schedule_callback=schedule_callback)
            gelu_patcher.patch(model)
            self.stats["gelu"] = gelu_patcher.stats

        return patcher


    def compile_model(self, model):
        self.schedule_threshold(compile=True)
        compiler = MaskedLinearModelCompiler()
        compiler.patch(model)

        if hasattr(self.sparse_args, "layer_norm_patch") and self.sparse_args.layer_norm_patch:
            nnc = NoNormCompiler()
            nnc.patch(model)
            model.config.layer_norm_type = "no_norm"

        if hasattr(self.sparse_args, "gelu_patch") and self.sparse_args.gelu_patch:
            model.config.hidden_act = "relu"

        pruner = BertHeadsPruner(model)
        removed_heads, total_heads = pruner.run()
        return removed_heads, total_heads
