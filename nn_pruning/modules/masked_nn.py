# coding=utf-8
# Copyright 2020-present, the HuggingFace Inc. team.
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
Masked Linear module: A fully connected layer that computes an adaptive binary mask on the fly.
The mask (binary or not) is computed at each forward pass and multiplied against
the weight matrix to prune a portion of the weights.
The pruned weight matrix is then multiplied against the inputs (and if necessary, the bias is added).
"""

from argparse import ArgumentError
import math
from dataclasses import dataclass
from itertools import permutations
from typing import List, Optional
from transformers.modeling_utils import Conv1D 

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from nn_pruning.model_structure import ModelStructure
from nn_pruning.model_patcher import ModelPatcher

from nn_pruning.training_patcher import (
    ModulePatcher,
    PatcherContext,
    PatcherContextModule,
    ReplacementModule,
)

from .binarizer import MagnitudeBinarizer, ThresholdBinarizer, TopKBinarizer,  SageBinarizer, GlobalTopKBinarizer, GroupMagnitudeBinarizer
import numpy

sparse_patterns = None

AMPERE_M = 4
AMPERE_N = 2

import torch.distributions


@dataclass
class RuntimeLinearPruningArgs:
    method: str
    submethod: str
    ampere_method: str
    block_rows: int
    block_cols: int
    min_elements: float
    bias_mask: bool = False

@dataclass
class InitDirective:
    kind: str = "constant"
    scale: float = 0.0


@dataclass
class LinearPruningArgs(RuntimeLinearPruningArgs):
    mask_init: InitDirective = InitDirective()
    ampere_init: InitDirective = InitDirective()
    sage_beta_meta: float = 0
    uniqueness_tau: float = 0
    soft_temperature: float = 0
    adjust_grad_lamb: float = 0
    mask_span_reg_lamb: float = 0
    scale_pruned: bool = False
    scale_fc: bool = False
    scale_proj: bool = False
    mask_frozen: bool = False


class GenericLinearPruningContextModule(PatcherContextModule):
    def __init__(self, shape, args: LinearPruningArgs):
        super().__init__()
        self.shape = shape
        self.args = args

    def init_masks(self, init_directive: InitDirective):
        mask_init = init_directive.kind
        mask_scale = init_directive.scale
        mask_scores = self.mask_scores
        if mask_init == "constant":
            init.constant_(mask_scores, val=mask_scale)
        elif mask_init == "uniform":
            init.uniform_(mask_scores, a=-mask_scale, b=mask_scale)
            print("using uniform random mask scores with scale", mask_scale)
        elif mask_init == "kaiming":
            init.kaiming_uniform_(mask_scores, a=math.sqrt(5))

        if self.args.mask_frozen:
            mask_scores.requires_grad = False
            print("mask scores are frozen - no grad")

    def regularization(self, method):
        mask_scores = self.mask_scores

        numel = mask_scores.numel()
        if method == "l1":
            if hasattr(self, "regu_mask_mult"):
                module_regu = torch.norm(torch.sigmoid(mask_scores)* self.regu_mask_mult, p=1) / numel
            elif hasattr(self, "threshold"):
                # module_regu = torch.norm(torch.sigmoid((mask_scores - self.threshold) / self.args.soft_temperature), p=1) / numel
                # module_regu = torch.sigmoid((mask_scores - self.threshold) / self.args.soft_temperature).sum() / numel
                
                module_regu = torch.relu((mask_scores - self.threshold) ).mean()
                # 0/0
                # module_regu = torch.norm(torch.sigmoid(mask_scores), p=1) / numel
            else:
                module_regu = torch.norm(torch.sigmoid(mask_scores), p=1) / numel
        elif method == "l0":
            module_regu = torch.sigmoid(mask_scores - 2 / 3 * numpy.log(0.1 / 1.1)).sum() / numel
        else:
            assert False

        return module_regu


class LinearPruningContextModule(GenericLinearPruningContextModule):
    def __init__(self, shape, args: LinearPruningArgs):
        super().__init__(shape, args)
        _p = self.args

        assert shape[0] % _p.block_rows == 0
        assert shape[1] % _p.block_cols == 0
        self.mask_size = (
            shape[0] // _p.block_rows,
            shape[1] // _p.block_cols,
        )


class BlockLinearPruningContextModule(LinearPruningContextModule):
    def __init__(self, shape, args: LinearPruningArgs):
        super().__init__(shape, args)
        assert args.submethod == "default"
        self.mask_scores = nn.Parameter(torch.Tensor(size=self.mask_size))
        self.init_masks(self.args.mask_init)

class SageLinearPruningContextModule(LinearPruningContextModule):
    def __init__(self, shape, args: LinearPruningArgs):
        super().__init__(shape, args)
        assert args.method == "sage"
        assert args.submethod == "default"
        self.mask_scores = nn.Parameter(torch.Tensor(size=self.mask_size))    # will need to exclude this from opt, don't use this at all
        # self.register_buffer('mask_scores', torch.Tensor(size=self.mask_size))
        self.register_buffer('ipt', torch.Tensor(size=self.mask_size))
        self.register_buffer('exp_avg_unc', torch.Tensor(size=self.mask_size))
        self.register_buffer('exp_avg_ipt', torch.Tensor(size=self.mask_size))
        self.init_masks(self.args.mask_init)


class SingleDimensionLinearPruningContextModule(LinearPruningContextModule):
    def __init__(self, shape, is_row, args: LinearPruningArgs):
        super().__init__(shape, args)
        assert args.submethod.startswith("1d")

        size = (self.mask_size[0],) if is_row else (self.mask_size[1],)

        self.mask_scores = nn.Parameter(torch.zeros(size=size))
        self.init_masks(self.args.mask_init)
        
        if self.args.adjust_grad_lamb != 0 or args.mask_span_reg_lamb != 0: # really should rename this now
            self.register_buffer('regu_mask_mult', torch.ones(size=size, requires_grad=False))

        if args.method == "uniqueness":
            print("Not optimizing mask scores, using uniqueness", "is row", is_row, size)
            self.mask_scores.requires_grad = False

        if args.soft_temperature > 0:
            self.threshold = nn.Parameter(self.mask_scores.new_zeros((1,)))
            # init.constant_(self.threshold, val=-1)
            init.constant_(self.threshold, val=0)




class AmpereLinearPruningContextModule(GenericLinearPruningContextModule):
    AMPERE_N = 2
    AMPERE_M = 4
    SPARSE_PATTERNS_COUNT = 6  # 2 choices among 4 = 6

    def __init__(self, shape, args: LinearPruningArgs):
        super().__init__(shape, args)
        if args.ampere_method == "sigmoied_threshold":
            ampere_mask_size = self.shape
        elif args.ampere_method == "annealing":
            # Creating the pattern in a transposed way to avoid a few ops later
            ampere_mask_size = (
                self.shape[1],
                self.shape[0] // self.AMPERE_M,
                self.SPARSE_PATTERNS_COUNT,
            )
        else:
            raise Exception(f"Unknown ampere method {args.ampere_method}")

        self.mask_scores = nn.Parameter(torch.Tensor(size=ampere_mask_size))
        self.init_masks(self.args.ampere_init)


class AmpereMaskModule(nn.Module):
    def __init__(
        self,
        context_module: AmpereLinearPruningContextModule,
        args: LinearPruningArgs,
    ):
        super().__init__()
        self.context_module = context_module
        ampere_pattern = self.build_ampere_pattern(context_module.mask_scores.device)
        self.register_buffer("ampere_pattern", ampere_pattern)
        self.args = args
        self.method = args.ampere_method

    @staticmethod
    def build_ampere_pattern(device):
        patterns = torch.zeros(AmpereLinearPruningContextModule.AMPERE_M)
        patterns[: AmpereLinearPruningContextModule.AMPERE_N] = 1
        sparse_patterns = torch.Tensor(list(set(permutations(patterns.tolist())))).to(device=device)
        return sparse_patterns

    @staticmethod
    def get_final_mask(mask_scores, pattern_n, pattern_m):
        assert (mask_scores.shape[1] % pattern_m) == 0

        mask_scores_m = mask_scores.reshape(mask_scores.shape[0], mask_scores.shape[1] // pattern_m, pattern_m)
        # print(f"ampere mask scores range [{mask_scores.min().item()}, {mask_scores.max().item()}]")
        top = torch.topk(mask_scores_m, pattern_n, dim=2, largest=True)
        top_mask = torch.zeros_like(mask_scores_m, device=mask_scores.device)
        top_mask = top_mask.scatter(2, top.indices, 1.0)
        top_mask = top_mask.reshape_as(mask_scores)

        return top_mask

    @staticmethod
    def sigmoied_threshold_mask(mask_scores, threshold, sigmoid, train):
        top_mask = AmpereMaskModule.get_final_mask(mask_scores, 2, 4)

        if train:
            mask = ThresholdBinarizer.apply(mask_scores, threshold, sigmoid, 0)
            ret = torch.max(mask, top_mask)
        else:
            ret = top_mask

        return ret

    @staticmethod
    def topk_mask(mask_scores, threshold, train):
        top_mask = AmpereMaskModule.get_final_mask(mask_scores, 2, 4)

        if train:
            mask = TopKBinarizer.apply(mask_scores, threshold)
            ret = torch.max(mask, top_mask)
        else:
            ret = top_mask

        return ret

    @staticmethod
    def annealing_mask(ampere_permut_scores, ampere_pattern, ampere_temperature: float, training: bool):
        if training:
            s = F.softmax(ampere_permut_scores * ampere_temperature, dim=-1)
        else:
            s = torch.argmax(ampere_permut_scores, dim=-1)
            s = F.one_hot(s, num_classes=ampere_permut_scores.shape[-1]).float()

        s = s.matmul(ampere_pattern)
        s = s.view(-1, s.shape[1] * s.shape[2])
        s = s.t()

        return s

    def forward(self, ampere_temperature):
        if self.method in ["threshold", "sigmoied_threshold"]:
            return self.sigmoied_threshold_mask(
                self.context_module.mask_scores,
                ampere_temperature,
                "sigmoied" in self.method,
                self.training,
            )
        elif self.method == "topK":
            return self.topk_mask(
                self.context_module.mask_scores,
                ampere_temperature,
                self.training,
            )
        elif self.method == "annealing":
            return self.annealing_mask(
                self.context_module.mask_scores,
                self.ampere_pattern,
                ampere_temperature,
                self.training,
            )
        else:
            raise Exception(f"Unknown ampere method {self.method}")

class MaskModule(nn.Module):
    def __init__(
        self,
        context_modules: List[LinearPruningContextModule],
        args: LinearPruningArgs,
    ):
        super().__init__()
        assert isinstance(context_modules, (list, tuple))
        self.context_modules = nn.ModuleList(context_modules)
        self.args = args

        if self.args.method == "sage":
            # only 1 module for sagepruning, structured not implemented yet
            # gets complicated passing the buffers into mask function if using multiple sage contexts
            assert len(context_modules) == 1 

    @staticmethod
    def expand_mask(mask, block_rows, block_cols):
        mask = torch.repeat_interleave(mask, block_rows, dim=0)
        mask = torch.repeat_interleave(mask, block_cols, dim=1)
        return mask

    @staticmethod
    def mask(
        weight,
        mask_scores,
        args: LinearPruningArgs,
        threshold: float,
        training,
        input,
        ipt=None,
        exp_avg_unc=None,
        exp_avg_ipt=None
    ):
        method = args.method
        if method == "disabled":
            return None

        submethod = args.submethod
        if submethod.startswith("1d"):
            dividers = args.block_rows, args.block_cols
            no_matmul = False
            for i, m in enumerate(mask_scores):
                if m is None:
                    # assert submethod == "alt"
                    assert "1d_alt" in submethod or "1d_only" in submethod
                    mask_scores[i] = torch.ones(weight.shape[i] // dividers[i], device=weight.device)

            if not no_matmul:
                mask_scores = mask_scores[0].unsqueeze(-1).matmul(mask_scores[1].unsqueeze(0))
            else:
                # mask_scores = mask_scores[0].unsqueeze(-1)
                mask_scores = mask_scores[0]
        elif method not in "magnitude":
            mask_scores = mask_scores[0]

        if method == "topK":
            mask = TopKBinarizer.apply(mask_scores, threshold)
        elif method == "magnitude":
            mask = MagnitudeBinarizer.apply(weight, threshold)
        elif method == "group_magnitude":
            mask = GroupMagnitudeBinarizer.apply(weight, threshold)
        elif method in ["threshold", "sigmoied_threshold"]:
            sig = "sigmoied" in method
            # print(threshold)
            mask = ThresholdBinarizer.apply(mask_scores, threshold, sig, args.min_elements)
        elif method in ["global_topK"]:
            mask = GlobalTopKBinarizer.apply(mask_scores, threshold, args.min_elements)
        elif method == "extra_soft":
            # print(threshold)
            # mask = (mask_scores - threshold) / args.soft_temperature
            # print(mask_scores.flatten()[:10])

            # mask = torch.sigmoid((mask_scores - threshold) / args.soft_temperature)
            mask = (mask_scores - threshold) / args.soft_temperature
            # print(mask_scores)
            # print(threshold)


            mask =  ThresholdBinarizer.apply(mask, 0, False, 0)
            # mask = mask * ThresholdBinarizer.apply(mask, .05, False, 0)
            # mask = ThresholdBinarizer.apply(mask, .1, False, 0)
            
            # mask = torch.sigmoid((torch.log(mask_scores) - threshold) / args.soft_temperature)
        elif method == "l0":
            l, r, b = -0.1, 1.1, 2 / 3
            if training:
                u = torch.zeros_like(mask_scores).uniform_().clamp(0.0001, 0.9999)
                s = torch.sigmoid((u.log() - (1 - u).log() + mask_scores) / b)
            else:
                s = torch.sigmoid(mask_scores)
            s_bar = s * (r - l) + l
            mask = s_bar.clamp(min=0.0, max=1.0)
        elif method == "sage":
            assert exp_avg_unc != None
            assert exp_avg_ipt != None

            mask = SageBinarizer.apply(ipt, exp_avg_unc, exp_avg_ipt, threshold, args.sage_beta_meta)
        else:
            raise NotImplementedError(f"Unknown method {method}")

        if method not in "magnitude":
            # Expand block mask to individual element mask
            mask = MaskModule.expand_mask(
                mask,
                block_rows=args.block_rows,
                block_cols=args.block_cols,
            )

        return mask

    def forward(self, weight, threshold, input):
        mask_scores = None
        if self.args.method != "magnitude":
            mask_scores = [(c.mask_scores if c is not None else None) for c in self.context_modules]

        
        if self.args.soft_temperature > 0:
            for c in self.context_modules:
                if c is not None:
                    threshold = c.threshold

        ipt = None
        exp_avg_unc = None
        exp_avg_ipt = None
        if self.args.method == "sage":
            # this is ugly, should not be done this way. Should refactor in the future
            sage_context = self.context_modules[0]
            ipt = sage_context.ipt
            exp_avg_unc = sage_context.exp_avg_unc
            exp_avg_ipt = sage_context.exp_avg_ipt  

        return self.mask(weight, mask_scores, self.args, threshold, self.training, input, ipt=ipt, exp_avg_unc=exp_avg_unc, exp_avg_ipt=exp_avg_ipt)

class MaskedLinear(ReplacementModule):
    def __init__(
        self,
        linear_module: nn.Linear,
        mask_context_modules: List[LinearPruningContextModule],
        ampere_context_module: AmpereLinearPruningContextModule,
        args: LinearPruningArgs,
        row_additive_mask: Optional[torch.Tensor]=None,
        col_additive_mask: Optional[torch.Tensor]=None
    ):
        super().__init__()

        assert isinstance(linear_module, nn.Linear) or isinstance(linear_module, Conv1D)
        self.weight = linear_module.weight if isinstance(linear_module, nn.Linear) else nn.Parameter(linear_module.weight.T.detach().clone())
        self.bias = linear_module.bias

        self.mask_module = MaskModule(mask_context_modules, args)
        if ampere_context_module is not None:
            self.ampere_module = AmpereMaskModule(ampere_context_module, args)
        self.args = args

        if row_additive_mask is not None:
            row_additive_mask = row_additive_mask.to(self.weight.device)
        self.row_additive_mask = row_additive_mask

        if col_additive_mask is not None:
            col_additive_mask = col_additive_mask.to(self.weight.device)

        self.col_additive_mask = col_additive_mask

        if self.args.method == "global_topK":
            self.register_buffer('global_topK_threshold', self.weight.new_zeros((1, )))
            # self.global_topK_threshold = 1

        self.is_scaling = False
        
        if self.args.scale_fc and self.weight.shape[1] < self.weight.shape[0]:  # this is fc because input < output
            print("using scale fc")
            self.is_scaling = True
            self.out_scale = nn.Parameter(self.weight.new_ones((self.weight.shape[0],)))

        if self.args.scale_proj and self.weight.shape[0] < self.weight.shape[1]:  # this is proj because output < input
            print("using scale proj")
            self.is_scaling = True
            self.out_scale = nn.Parameter(self.weight.new_ones((self.weight.shape[0],)))


    def nnz(self, m):
        if len(m.shape) == 1:
            return int((m != 0).sum().item()*self.weight.shape[1])
        else:
            return int((m != 0).sum().item())

    def get_masked_weights_bias(self, input=None):
        # if self.args.soft_temperature > 0:
        #     threshold = self.threshold
        # else:

        if self.args.method == "global_topK":
            threshold = self.global_topK_threshold
        else:
            threshold = self.get_context_data("threshold")

        weights = self.weight

        mask = self.mask_module(weights, threshold, input)

        if mask is not None:
            if self.row_additive_mask is not None:
                row_mask = self.row_additive_mask
                row_mask = row_mask.unsqueeze(-1)
                row_mask = row_mask.expand_as(mask).float()
                mask = torch.maximum(mask, row_mask)
            if self.col_additive_mask is not None:
                col_mask = self.col_additive_mask
                col_mask = col_mask.expand_as(mask).float()
                mask = torch.maximum(mask, col_mask)

        if mask is not None:
            self.mask_nnz = self.nnz(mask)
        else:
            self.mask_nnz = weights.numel()

        if self.args.ampere_method != "disabled":
            ampere_temperature = self.get_context_data("ampere_temperature")
            ampere_mask = self.ampere_module(ampere_temperature)
            if mask is not None and mask.shape != ampere_mask.shape:
                raise Exception("Shape mismatch")
            self.ampere_nnz = self.nnz(ampere_mask)
            if mask is not None:
                mask = mask * ampere_mask
            else:
                mask = ampere_mask
            self.base_mask_nnz = self.mask_nnz
            self.mask_nnz = self.nnz(mask)
        
        if mask is not None:
            masked_weights = mask * weights
        else:
            masked_weights = weights

        bias = self.bias
        if bias is not None:
            if self.args.bias_mask and mask is not None:
                bias = bias * (mask != 0).any(1)


        if self.is_scaling:
            masked_weights = masked_weights * self.out_scale.view((-1, 1))
            bias = bias * self.out_scale

        return masked_weights, bias, threshold, mask

    def forward(self, input):
        # masked_weights, bias = self.get_masked_weights_bias(input)
        # Compute output (linear layer) with masked weights
        # return F.linear(input, masked_weights, bias)
        
        masked_weights, bias, threshold, mask = self.get_masked_weights_bias(input)

        if self.args.scale_pruned and masked_weights.shape[0] < masked_weights.shape[1]: # so this should be c_proj only?
            # print (masked_weights.shape[1] , inputs_leftover)
            # output *= (masked_weights.shape[1] / inputs_leftover)
            # print (threshold , self.mask_nnz / self.weight.numel())
            # masked_weights *= threshold**-.5

            # masked_weights *= (self.mask_nnz / self.weight.numel())**-.5

            masked_weights = masked_weights / threshold


        output = F.linear(input, masked_weights, bias)


        # inputs_leftover = mask.sum(axis=1)[0].detach()

        # if self.is_scaling:
        #     output = output * self.out_scale

        # return output, gated, mask
        return output


    def get_sparsity_info(self):
        masked_weights, bias, threshold, mask = self.get_masked_weights_bias()   # need this to set mask_nnz... gross
        ret = {"numel": self.weight.numel(), "nnz": self.mask_nnz}

        if self.args.ampere_method != "disabled":
            ret.update({"base_nnz": self.base_mask_nnz, "ampere_nnz": self.ampere_nnz})
        return ret

    def compile(self):
        masked_weights, bias, threshold, mask  = self.get_masked_weights_bias()

        ret = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=self.bias is not None)
        with torch.no_grad():
            ret.weight.copy_(masked_weights)
            if ret.bias is not None:
                ret.bias.copy_(bias)
        return ret

class LinearPruningModulePatcher(ModulePatcher):
    def __init__(
        self,
        context: PatcherContext,
        args: LinearPruningArgs,
        model_structure:ModelStructure,
        row_additive_mask: Optional[torch.Tensor] = None,
        col_additive_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__(context=context)
        self.args = args
        self.check_method(args)
        self.model_structure = model_structure
        self.row_additive_mask = row_additive_mask
        self.col_additive_mask = col_additive_mask

    def layer_index(self, child_module_name):
        return self.model_structure.layer_index(child_module_name)

    @staticmethod
    def check_method(args: LinearPruningArgs):
        method = args.method
        submethod = args.submethod
        ampere_method = args.ampere_method
        PRUNING_METHODS = ["disabled", "topK", "threshold", "sigmoied_threshold", "magnitude", "l0",  "sage", "extra_soft", "global_topK", "group_magnitude"]
        if method not in PRUNING_METHODS:
            raise RuntimeError(f"Unknown pruning method '{method}', should be in {PRUNING_METHODS}")

        PRUNING_SUB_METHODS = ["default", "1d", "1d_alt","1d_alt_plus","1d_only", "joint"]
        if submethod not in PRUNING_SUB_METHODS:
            raise RuntimeError(f"Unknown pruning sub method '{submethod}', should be in {PRUNING_SUB_METHODS}")

        AMPERE_METHODS = ["disabled", "annealing", "sigmoied_threshold", "topK"]
        if ampere_method not in AMPERE_METHODS:
            raise RuntimeError(f"Unknown ampere pruning method '{ampere_method}', should be in {AMPERE_METHODS}")

    def create_context_module(self, child_module_name, child_module, key, row=None):
        shape = child_module.weight.shape if isinstance(child_module, nn.Linear) else child_module.weight.T.shape
        args = self.args
        prefix = key[0]
        if prefix == "mask":
            assert args.submethod in ["default", "joint"]
            if args.method == "sage":
                return SageLinearPruningContextModule(shape, args)
            return BlockLinearPruningContextModule(shape, args)
        elif prefix == "mask_1d":
            assert args.submethod.startswith("1d")
            assert row in [True, False]
            return SingleDimensionLinearPruningContextModule(shape, row, args)
        elif prefix == "ampere_mask":
            return AmpereLinearPruningContextModule(shape, args)
        else:
            raise RuntimeError(f"Unknown context module kind {prefix}")

    def patch(self, child_module_name, child_module):
        if self.args.method != "disabled":
            if self.args.submethod.startswith("1d"):
                mask_row = self.get_context_module(child_module_name, child_module, kind="mask_row", row=True)
                mask_col = self.get_context_module(child_module_name, child_module, kind="mask_col", row=False)
                shape = child_module.weight.shape  if isinstance(child_module, nn.Linear) else child_module.weight.T.shape
                if mask_row is not None:
                    assert mask_row.mask_scores.shape[0] == shape[0] // self.args.block_rows
                if mask_col is not None:
                    assert mask_col.mask_scores.shape[0] == shape[1] // self.args.block_cols
                mask_context_modules = [mask_row, mask_col]
            else:
                mask_context_module = self.get_context_module(child_module_name, child_module, kind="mask")
                mask_context_modules = [mask_context_module]
        else:
            mask_context_modules = []

        if self.args.ampere_method != "disabled":
            ampere_context_module = self.get_context_module(child_module_name, child_module, kind="ampere_mask")
        else:
            ampere_context_module = None

        layer_index = self.layer_index(child_module_name)

        if self.row_additive_mask is not None:
            row_additive_mask = self.row_additive_mask[layer_index]
        else:
            row_additive_mask = None

        if self.col_additive_mask is not None:
            col_additive_mask = self.col_additive_mask[layer_index]
        else:
            col_additive_mask = None

        return MaskedLinear(child_module, mask_context_modules, ampere_context_module, self.args,
                            row_additive_mask = row_additive_mask,
                            col_additive_mask = col_additive_mask)


class JointPruningModulePatcher(LinearPruningModulePatcher):
    def __init__(self, context: PatcherContext, args: LinearPruningArgs, model_structure: ModelStructure, suffix: str):

        super().__init__(context, args, model_structure=model_structure)
        self.suffix = suffix

    def get_context_key(self, child_module_name, kind="default"):
        if kind == "ampere_mask":
            return (kind, child_module_name)
        elif kind == "mask":
            layer_number = self.layer_index(child_module_name)
            network_component = 'decoder' if self.model_structure.is_decoder(child_module_name) else 'encoder'
            return (kind, f"{network_component}.{layer_number}.{self.suffix}")
        else:
            raise RuntimeError(f"Unknown kind {kind}")


class ChannelPruningModulePatcher(LinearPruningModulePatcher):
    def __init__(
        self,
        context: PatcherContext,
        args: LinearPruningArgs,
        model_structure: ModelStructure,
        suffix: str,
    ):
        super().__init__(context, args, model_structure=model_structure)
        self.suffix = suffix

    def get_context_key(self, child_module_name, kind="default"):
        if kind == "ampere_mask":
            return (kind, child_module_name)
        elif kind in ["mask_row", "mask_col"]:
            layer_number = self.layer_index(child_module_name)

            offset = 1 if kind == "mask_row" else 0  # The weight matrix has a shape [output, input]

            # position, name = self.model_structure.get_module_intra_layer_position(child_module_name)
            '''
            I have no clue how this is supposed to work. c_fc has an even position and c_proj hAs an odd one?
            this is so unsafe and doesn't work for gpt2
            because this depends on attention heads being their own matrices, so they can be in layer patterns, which gpt2 doesn't do
            so, the positions are messed up for gpt2 and both cproj and cfc get masks...
            For now, just hardcode the below ifs
            '''
            if "c_fc" in child_module_name:
                position = 0
            elif "c_proj"  in child_module_name:
                position = 1
            else:
                position, name = self.model_structure.get_module_intra_layer_position(child_module_name)




            if self.args.submethod == "1d_alt":
                if (position % 2) == 1:
                    if kind == "mask_row":
                        return None
                else:
                    if kind == "mask_col":
                        return None

            if self.args.submethod == "1d_alt_plus": # only don't prune inputs for d -> 4d
                if (position % 2) == 0:
                    if kind == "mask_col":
                        return None

            if self.args.submethod == "1d_only": # prune only d dimension row=output, col=input, position 1 = output
                if (position % 2) == 0:
                    # if kind == "mask_row":  # DONT mask position 0 output, position 1 input
                    #     return None

                    return None  # for now, don't mask anything for position 0 (try inputs pruning later)
                else:
                    if kind == "mask_col":
                        return None

            network_component = 'decoder' if self.model_structure.is_decoder(child_module_name) else 'encoder'
            return ("mask_1d", f"{network_component}.{layer_number}.{position+offset}.{self.suffix}")
        else:
            raise RuntimeError(f"Unknown kind {kind}")


class MaskedLinearModelCompiler(ModelPatcher):
    def __init__(self):
        super().__init__(all_match=True)

    def is_patchable(self, module_name, module, raiseError):
        return isinstance(module, MaskedLinear)

    def new_child_module(self, child_module_name, child_module, patch_info):
        return child_module.compile()



def head_mask(config, device):
    removed_heads = config.pruned_heads
    num_attention_heads = config.num_attention_heads
    num_hidden_layers = config.num_hidden_layers

    head_keep = []
    head_size = int(config.hidden_size / num_attention_heads)

    for hidden_index in range(num_hidden_layers):
        r = removed_heads.get(hidden_index, [])
        keep = []
        for i in range(num_attention_heads):
            if i not in r:
                value = True
            else:
                value = False

            for j in range(head_size):
                keep.append(value)

        t = torch.tensor(keep, device=device, dtype=torch.bool)
        head_keep.append(t)
    return head_keep
