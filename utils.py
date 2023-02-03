from collections import defaultdict
from re import S
import torch
import os
from transformers import Trainer, TrainerCallback
from transformers.trainer_pt_utils import nested_detach
from nn_pruning.sparse_trainer import SparseTrainer, TimingModule
import torch 
# from transformers import AutoModelForCausalLM, AutoConfig
# from transformers import AutoConfig
from nn_pruning.patch_coordinator import ModelPatchingCoordinator
from nn_pruning.inference_model_patcher import optimize_model
from nn_pruning.modules.masked_nn import MaskedLinear, SageLinearPruningContextModule, SingleDimensionLinearPruningContextModule
from model import GPTNeoMLP, GPT2MLP, GPTNeoBlock
import numpy as np
import copy
from torch import nn
from transformers.optimization import AdamW
from nn_pruning.patch_coordinator import SparseTrainingArguments



def args_to_hparams(args):
    sparse_args = SparseTrainingArguments()

    initial_threshold = 1.0
    final_threshold = args.prune_leftover
    if "threshold" in args.dense_pruning_method or "sigmoied_threshold" in args.dense_pruning_method or "extra_soft" in args.dense_pruning_method:
        # different meaning for threshold, this is the bar to pass for mask scores
        print("setting init threshold 0")
        initial_threshold = 0 
        final_threshold = .1 

    regularization_final_lambda = 0
    if args.regularization != "disabled":
        regularization_final_lambda = args.regu_lamb
    
    weight_regularization_final_lambda = 0
    if args.weight_regularization != "disabled":
        weight_regularization_final_lambda = args.weight_regu_lamb

    linear_min_parameters = args.prune_leftover
    if args.dense_pruning_method == "global_topK":
        # linear_min_parameters = args.prune_leftover / 2
        linear_min_parameters = 0

    hyperparams = {
        "seed": args.seed,
        "dense_pruning_method": args.dense_pruning_method + ":" + args.dense_pruning_submethod, 
        "attention_pruning_method": args.attention_pruning_method, 
        "regularization": args.regularization,
        "weight_regularization": args.weight_regularization,
        "regularization_final_lambda": regularization_final_lambda,
        "weight_regularization_final_lambda": weight_regularization_final_lambda,
        "ampere_pruning_method": "disabled",
        "initial_threshold": initial_threshold, 
        "final_threshold": final_threshold, 
        # "initial_warmup": 1,
        "initial_warmup": args.initial_warmup,
        "final_warmup": 3,
        "attention_block_rows":32,
        "attention_block_cols":32,
        # "attention_block_rows":1,
        # "attention_block_cols":1,
        "attention_output_with_dense": 0,
        "schedule_type": args.schedule,
        "linear_min_parameters": linear_min_parameters,
        "mask_init": args.mask_init,
        "mask_scale": args.mask_init_scale,
        "mask_frozen": args.mask_frozen,
        # "mask_scores_learning_rate": 100,
        
        "sage_delta_T": args.sage_delta_T,
        "sage_beta_3": args.sage_beta_3,
        "sage_beta_meta": args.sage_beta_meta,
        "zero_pruned": args.zero_pruned,
        
        "train_only_bias_ln": args.train_only_bias_ln,

        "span_reg_lamb": args.span_reg_lamb,
        "A_reg_lamb": args.A_reg_lamb,
        "mask_span_reg_lamb": args.mask_span_reg_lamb,
        "opt_span_reg_only": args.opt_span_reg_only,
        "span_reg_A_learning_rate": args.span_reg_A_learning_rate,
        "running_r2_mult": args.running_r2_mult,

        "cpu_cos_sim": args.cpu_cos_sim,
        "adjust_grad_lamb": args.adjust_grad_lamb,
        "running_cos_method": args.running_cos_method,
        "running_cos_mult": args.running_cos_mult,
        "track_eval_cos": args.track_eval_cos,

        "scale_pruned": args.scale_pruned,
        "scale_fc": args.scale_fc,
        "scale_proj": args.scale_proj,
        "scale_params_learning_rate": args.scale_params_learning_rate,
        "uniqueness_reg_mask": args.uniqueness_reg_mask,
        "adjust_grad_do_mult": args.adjust_grad_do_mult,
        "adjust_mask_grad": args.adjust_mask_grad,
        "anti_gum": args.anti_gum,

        "soft_temperature": args.soft_temperature,

        "distil_teacher_name_or_path": args.distil_teacher_name_or_path,
        "distil_alpha_ce": args.distil_alpha_ce,
        "distil_alpha_teacher": args.distil_alpha_teacher,
        "distil_temperature": args.distil_temperature,
    }
    
    if "topK" in args.dense_pruning_method or "threshold" in args.dense_pruning_method or "sigmoied_threshold" in args.dense_pruning_method:
        hyperparams["mask_scores_learning_rate"] = args.mask_lr

    for k,v in hyperparams.items():
        if hasattr(sparse_args, k):
            setattr(sparse_args, k, v)
        else:
            print(f"sparse_args does not have argument {k}")
    
    return sparse_args



@torch.no_grad()
def adjust_grad(cos_sims, args, prune_threshold, model, output, adjust_grad_lamb, print_stats=False):

    all_sim = []
    all_neuron_scores = []
    all_neuron_multipliers = []
    # cnt = 0
    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
        running_c_fc_norm = cos_sims[n + "running_c_fc_norm"] 
        running_c_fc_sum =  cos_sims[n + "running_c_fc_sum"] 

        norm = running_c_fc_norm ** .5
        norm = norm.view(-1, 1)
        sim = running_c_fc_sum /  (  norm * norm.T + 1e-6   )
        all_sim.append(sim.abs() ) 


    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    
    for n, block in enumerate(model.transformer.h):
        sim = all_sim[n]


        # if args.adjust_grad_agg_method == "pos_perc":
        idx = range(0, sim.shape[1])
        sim[idx, idx] = 0 # so they get ignored
        # if print_stats:
        #     print("PRINTING SIM STATS")  # hopefully these go down over time...
            # print(sim)
            # print(sim.max(), sim.mean())

        # cutoff = .9
        # pos_neuron_scores = (sim > cutoff).sum(axis=0).to(sim.dtype)

        # pos_neuron_scores = sim.sum(axis=0).to(sim.dtype)

        
        pos_neuron_scores = sim.abs().sum(axis=0).to(sim.dtype)


        # pos_neuron_scores = (sim**2).sum(axis=0).to(sim.dtype)

        # cutoff = .95
        # cutoff = (1-prune_threshold) *.1      + .86
        # cutoff = (1-prune_threshold) *.08      + .9
        # pos_neuron_scores = ((sim > cutoff)**2).sum(axis=0).to(sim.dtype)
        # neg_neuron_scores = (sim < .1).sum(axis=1).to(sim.dtype)

       
        if print_stats:
            print("PRINTING pos_neuron_scores SCORE STATS")  
            print(pos_neuron_scores.max(), pos_neuron_scores.mean(),  pos_neuron_scores.sum())

        # pos_neuron_scores /= (sim.shape[0])
        pos_neuron_scores /= (sim.shape[0]*prune_threshold)
        multiplier = pos_neuron_scores*adjust_grad_lamb
        if args.anti_gum:
            mutliplier = torch.max(multiplier) - multiplier
            if print_stats:
                print("USING ANTI GUM...")
        if print_stats:
            print("PRINTING multiplier STATS")
            print(multiplier.max(), multiplier.mean(), "prune_threshold", prune_threshold, "do mult", args.adjust_grad_do_mult, "reg mask", args.uniqueness_reg_mask)

        
        if args.adjust_grad_do_mult:
            # 0/0
            block.mlp.c_fc.weight.grad += block.mlp.c_fc.weight.grad * (multiplier.view(-1, 1))
            # block.mlp.c_proj.weight.grad += block.mlp.c_proj.weight.grad * (multiplier.view(1, -1))

            # block.mlp.c_fc.weight.grad[unmod_idx]  *= (old*block.mlp.c_fc.weight.grad.shape[0]  - block.mlp.c_fc.weight.grad[mod_idx].mean(axis=1).sum()) /  block.mlp.c_fc.weight.grad[unmod_idx].sum(axis=0)

            # old = block.mlp.c_fc.weight.grad.mean()
            # block.mlp.c_fc.weight.grad += block.mlp.c_fc.weight.grad * (multiplier.view(-1, 1))
            # block.mlp.c_fc.weight.grad -= (block.mlp.c_fc.weight.grad.mean() - old)
            

        if args.uniqueness_reg_mask:
            # 0/0
            # mask_scores = block.mlp.c_fc.mask_module.context_modules[0].mask_scores
            # mask_scores -= multiplier*mask_scores

            # mask_scores.grad -= multiplier*mask_scores.grad

            if args.cpu_cos_sim:
                multiplier = multiplier.to(block.mlp.c_fc.mask_module.context_modules[0].regu_mask_mult.device)
            
            block.mlp.c_fc.mask_module.context_modules[0].regu_mask_mult[:] = multiplier + 1


        if args.adjust_mask_grad:
            if print_stats:
                print("doing adjust_mask_grad")
            mask_scores = block.mlp.c_fc.mask_module.context_modules[0].mask_scores

            mult =  multiplier / mask_scores.grad.norm() 

            mask_scores.grad -= mult







    return all_sim, all_neuron_scores, all_neuron_multipliers


@torch.no_grad()
def update_current_total_cos(cos_sims, args, model, output):
    all_neuron_outputs_fc = output["all_neuron_outputs_fc"]
    # all_neuron_outputs_proj = output["all_neuron_outputs_proj"]

    cnt = 0
    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
        out = all_neuron_outputs_fc[cnt].detach().clone()
        out.view((-1, out.shape[-1]))

        out_corr = out.T @ out
        out_norm = out.norm(dim=0) **2

        
        # cos_sims[n + "eval_c_fc_sum"]  =  cos_sims[n + "eval_c_fc_sum"] + out_corr     # update whether or not we are really evaluating, there is not an easy way to tell
        # cos_sims[n + "eval_c_fc_norm"]  = cos_sims[n + "eval_c_fc_norm"] + out_norm

        # mask = mod.c_fc.get_mask()[:, 0] > 0
        # cos_sims[n + "mask"] = mask.clone().detach()

        cos_sims[n + "current_c_fc_sum"]  = out_corr.to("cpu") if args.cpu_cos_sim else out_corr
        cos_sims[n + "current_c_fc_norm"]  = out_norm.to("cpu") if args.cpu_cos_sim else out_norm


        cnt += 1


@torch.no_grad()
def update_running_cos(cos_sims, args, model, output):
    # all_neuron_outputs_fc = output["all_neuron_outputs_fc"]
    # all_neuron_outputs_proj = output["all_neuron_outputs_proj"]
    # parallel = False
    # if isinstance(model, torch.nn.DataParallel):
    #     parallel = True


    current_cos = {}

    cnt = 0
    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
        # out = all_neuron_outputs_fc[cnt]
        # out.view((-1, out.shape[-1]))
        # out_corr = out.T @ out
        # out_norm = out.norm(dim=0) **2
        
        out_corr = cos_sims[n + "current_c_fc_sum"] 
        out_norm = cos_sims[n + "current_c_fc_norm"] 

        # mask = cos_sims[n + "mask"] 

        if not ((n + "running_c_fc_sum") in cos_sims.keys()): 
            # cos_sims[n + "running_c_fc_sum"] =  args.running_cos_mult * out_corr[mask, :]
            # cos_sims[n + "running_c_fc_norm"] = args.running_cos_mult * out_norm[mask]
            cos_sims[n + "running_c_fc_sum"] =  (args.running_cos_mult * out_corr)
            cos_sims[n + "running_c_fc_norm"] = (args.running_cos_mult * out_norm)
        elif True:
            cos_sims[n + "running_c_fc_sum"] = ((1-args.running_cos_mult)*cos_sims[n + "running_c_fc_sum"] + args.running_cos_mult * out_corr)
            cos_sims[n + "running_c_fc_norm"] = ((1-args.running_cos_mult)*cos_sims[n + "running_c_fc_norm"] + args.running_cos_mult * out_norm)
        else:
            # this will NOT update anything that has been pruned
            #however, note this freezes their cosine similarity... so grad adjust just keeps happening. not good.

            # before = cos_sims[n + "running_c_fc_sum"].detach().clone()

            running_c_fc_sum_orig = cos_sims[n + "running_c_fc_sum"] 
            antimask = mask == False

            running_c_fc_sum = cos_sims[n + "running_c_fc_sum"] 
            running_c_fc_sum *=   (1-args.running_cos_mult)
            running_c_fc_sum +=  args.running_cos_mult * out_corr[mask, :]

            running_c_fc_sum[antimask, :] = running_c_fc_sum_orig[antimask, :]
            running_c_fc_sum[ :, antimask] = running_c_fc_sum_orig[ :, antimask]


            running_c_fc_norm =  cos_sims[n + "running_c_fc_norm"] 
            running_c_fc_norm[mask] *=   (1-args.running_cos_mult)
            running_c_fc_norm[mask] += args.running_cos_mult * out_norm[mask]

            cos_sims[n + "running_c_fc_sum"] = running_c_fc_sum
            cos_sims[n + "running_c_fc_norm"] = running_c_fc_norm

            # print((cos_sims[n + "running_c_fc_sum"] - before).abs().sum())


        del cos_sims[n + "current_c_fc_sum"] 
        del cos_sims[n + "current_c_fc_norm"] 

        cnt += 1


@torch.no_grad()
def distribute_thresholds(args, model, threshold):
    scores = []
    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
        if not isinstance(mod.c_fc, MaskedLinear): continue
        scores.append(mod.c_fc.mask_module.context_modules[0].mask_scores.flatten())
        
    if len(scores) == 0: return

    all_scores = torch.cat(scores)
    all_scores, idx = all_scores.flatten().sort(descending=True)
    j = int(threshold * (all_scores.numel()) ) - 1 
    # if getting weird results - I messed with this by accident, it should be reverted
    # but why -1 here? seems like the wrong way to do thigns... need to investigate


    threshold_elem = all_scores[j]
    # print(threshold,  all_scores[j],  all_scores[0])

    # print((all_scores >= threshold_elem).sum(), all_scores.numel())

    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
        # print(threshold_elem)
        mod.c_fc.global_topK_threshold[:] = threshold_elem
        mod.c_proj.global_topK_threshold[:] = threshold_elem


@torch.no_grad()
def print_gated_input(model, output):
    all_gated_fc_input = output["all_gated_fc_input"]
    all_gated_proj_input = output["all_gated_proj_input"]
    # all_neuron_outputs_proj = output["all_neuron_outputs_proj"]


    print("PRINTING GATED INPUT COUNTS")
    cnt = 0 
    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
        gated_fc_input = all_gated_fc_input[cnt].detach().clone()
        gated_proj_input = all_gated_proj_input[cnt].detach().clone()

        bincount = torch.bincount(gated_fc_input)
        print(n, "fc", bincount)
        bincount = torch.bincount(gated_proj_input)
        print(n, "proj", bincount)


        cnt += 1

def init_span_reg(model):
    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
        mod.init_span_reg()

def get_A_reg_l1(model):
    reg = 0
    numel = 0
    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
        # mod.init_span_reg()
    # for A in self.As:
        # A = mod.span_reg_A
        reg += mod.span_reg_A.abs().sum()
        numel += mod.span_reg_A.numel()
    return reg / numel

@torch.no_grad()
def udpate_span_regs(span_regs, args, model, output, print_stats):

    r2s = []
    cnt = 0
    total_sim_gt9 = 0
    total_sim_r2_gt9 = 0
    total_nnz_neurons = 0
    total_sim_gt99 = 0
    total_sim_r2_gt99 = 0
    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
        # don't technically need to do this, could get it elsewhere faster. But, want to make sure this is 100% correct
        masked_weights, bias, threshold, mask = mod.c_fc.get_masked_weights_bias() 
        total_nnz_neurons += mask[:, 0].sum()


        out_span_reg = output["all_span_reg"][cnt].detach().clone()
        out_span_h = output["all_neuron_outputs_fc"][cnt].detach().clone()
        

        # span_regs[n + "running_span_reg"] = (1-args.running_cos_mult)*span_regs[n + "running_span_reg"] + args.running_cos_mult * out_span_reg
        # span_regs[n + "running_span_hidden"] = (1-args.running_cos_mult)*span_regs[n + "running_span_hidden"] + args.running_cos_mult * out_span_h
        
        span_regs[n + "running_span_reg"] =  out_span_reg
        span_regs[n + "running_span_hidden"] =  out_span_h
        
        s = span_regs[n + "running_span_reg"]
        h = span_regs[n + "running_span_hidden"]
        # span_r2_per_neuron = 1 - ((s**2).sum(dim=0) / ((h - h.mean(dim=0))**2).sum(dim=0)  ).flatten()
        # span_r2_per_neuron = 1 - ((s**2).mean(dim=0) / ((h - h.mean(dim=0))**2).mean(dim=0)  ).flatten()
        span_r2_per_neuron = 1 - ((s**2).mean(dim=0) / ((h)**2).mean(dim=0)  ).flatten()

        
        span_r2_per_neuron = torch.nan_to_num(span_r2_per_neuron, posinf=0, neginf=0)
        span_r2_per_neuron[span_r2_per_neuron < 0] = 0 # comes from h being exactly the mean, just set to 0. this is very rare anyway.
        # r2 = span_r2_per_neuron * mask if mask is not None else span_r2_per_neuron
        r2 = span_r2_per_neuron 
        r2s.append(r2.detach().clone())

        sim = h.T @ h 
        norm = h.norm(dim=0) .view(-1, 1)
        sim = sim /  (  norm * norm.T + 1e-6   )
        idx = range(0, sim.shape[1])
        sim[idx, idx] = 0 # so they get ignored
        
        # cos_gt9 = ((sim > .9).sum(axis=1) > 0).to(torch.int) * mask if mask is not None else span_r2_per_neuron
        cos_gt9 = ((sim > .9).sum(axis=1) > 0).to(torch.int) 
        cos_gt99 = ((sim > .99).sum(axis=1) > 0).to(torch.int) 


        total_sim_gt9 += cos_gt9.sum()
        total_sim_r2_gt9 += cos_gt9[r2>.90].sum()
        total_sim_gt99 += cos_gt99.sum()
        total_sim_r2_gt99 += cos_gt99[r2>.99].sum()

        if print_stats:
            nz = r2[r2!=0]
            print("span reg", n, "mean", r2.mean(), "max", r2.max(), "nzmin", nz.min() if nz.numel() > 0 else None, "var", r2.var())
            print("span reg", n, ">.99", (r2>.99).sum(), ">.95", (r2>.95).sum(), ">.90", (r2>.90).sum())
            print("cos_gt9", cos_gt9.sum(), "sim_r2_gt9", cos_gt9[r2>.90].sum(), ">.90", (r2>.90).sum() )
            print("cos_gt99", cos_gt99.sum(), "sim_r2_gt99", cos_gt9[r2>.99].sum(), ">.99", (r2>.99).sum() )
            
            # masked_weights, bias, threshold, mask = mod.c_fc.get_masked_weights_bias()
            # mask = mask[:, 0]
            # print("misantac debug masked r2 > .9 ",  ((r2>.90).to(torch.int)  * (mask==0)).sum() )
            # print("misantac debug masked cossim > .9 ",  (cos_gt9 * (mask==0)).sum() )
            # print("misantac test debug nnz mask", mask.sum() / mask.numel(), mask.numel(), mask.sum())
        cnt += 1

    r2s_calc = torch.cat(r2s)

    stats = {}
    stats["r2s"] = r2s
    stats["mean"] = r2s_calc.sum() / total_nnz_neurons
    stats["gt99"] = (r2s_calc>.99).sum() / total_nnz_neurons
    stats["gt95"] = (r2s_calc>.95).sum() / total_nnz_neurons
    stats["gt90"] = (r2s_calc>.90).sum() / total_nnz_neurons
    stats["sim_gt90"] = total_sim_gt9 / total_nnz_neurons
    stats["perc_cos_also_r2_gt90"] =  total_sim_r2_gt9 / total_sim_gt9
    stats["perc_cos_also_r2_gt99"] =  total_sim_r2_gt99 / total_sim_gt99

    return stats

@torch.no_grad()
def do_mask_span_reg_lamb(r2s, args, model, mask_span_reg_lamb, print_stats):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for n, block in enumerate(model.transformer.h):
        r2 = r2s[n]
        # r2[r2 < .9] = 0
        r2[r2 < .99] = 0
        mult = mask_span_reg_lamb*r2 + 1
        block.mlp.c_fc.mask_module.context_modules[0].regu_mask_mult[:] = mult

        if print_stats:
            print("multiplying", n, mult.mean(), mult.max(), mult.min())



@torch.no_grad()
def get_r2_stats(model, outputs, total_nnz_neurons):
    # total_norm = 0
    r2s = []
    cnt = 0
    total_sim_gt9 = 0
    total_sim_r2_gt9 = 0
    total_sim_gt99 = 0
    total_sim_r2_gt99 = 0
    total_lt_0 = 0

    total_num_contributors = 0
    total_neurons = 0
    total_A_numel = 0
    total_A_gt1 = 0

    total_num_contributors_gt99 = 0
    total_r2_gt99 = 0

    cnt = 0
    for n, mod in model.named_modules():
        if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue

        A = mod.span_reg_A
        span_r2_per_neuron = outputs["all_span_r2"][cnt].reshape(-1, A.shape[0]).mean(dim=0)
        cnt += 1

        # print( outputs["all_span_r2"])
        # print(span_r2_per_neuron.shape)
        # 0/0
        
        total_lt_0 += (span_r2_per_neuron < 0).sum()
        span_r2_per_neuron[span_r2_per_neuron < 0] = 0 # REMOVE LESS THAN 0 
        # r2 = span_r2_per_neuron * mask if mask is not None else span_r2_per_neuron
        r2 = span_r2_per_neuron 
        r2s.append(r2.detach().clone())

        
        ct = 1e-4
        total_num_contributors += (A.abs() > ct).sum()
        total_neurons += int(A.shape[0])
        total_A_numel += A.numel()
        total_A_gt1 += (A.abs() > ct).sum()

        r2_gt99 = r2>.99
        total_num_contributors_gt99 += (A.abs() > ct).sum(axis=0)[r2_gt99].sum()
        total_r2_gt99 += r2_gt99.sum()



        # sim = h.T @ h 
        # norm = h.norm(dim=0) .view(-1, 1)
        # sim = sim /  (  norm * norm.T + 1e-6   )
        # idx = range(0, sim.shape[1])
        # sim[idx, idx] = 0 # so they get ignored
        # cos_gt9 = ((sim > .9).sum(axis=1) > 0).to(torch.int) 
        # cos_gt99 = ((sim > .99).sum(axis=1) > 0).to(torch.int) 

        # total_sim_gt9 += cos_gt9.sum()
        # total_sim_r2_gt9 += cos_gt9[r2>.90].sum()
        # total_sim_gt99 += cos_gt99.sum()
        # total_sim_r2_gt99 += cos_gt99[r2>.99].sum()

        # if print_stats:
        #     nz = r2[r2!=0]
        #     print("span reg", n, "mean", r2.mean(), "max", r2.max(), "nzmin", nz.min() if nz.numel() > 0 else None, "var", r2.var())
        #     print("span reg", n, ">.99", (r2>.99).sum(), ">.95", (r2>.95).sum(), ">.90", (r2>.90).sum())
        #     print("cos_gt9", cos_gt9.sum(), "sim_r2_gt9", cos_gt9[r2>.90].sum(), ">.90", (r2>.90).sum() )
        #     print("cos_gt99", cos_gt99.sum(), "sim_r2_gt99", cos_gt9[r2>.99].sum(), ">.99", (r2>.99).sum() )
        # cnt += 1

    r2s_calc = torch.cat(r2s)

    stats = {}
    stats["r2s"] = r2s
    stats["mean"] = r2s_calc.sum() / total_nnz_neurons
    stats["gt99"] = (r2s_calc>.99).sum() / total_nnz_neurons
    # stats["gt95"] = (r2s_calc>.95).sum() / total_nnz_neurons
    stats["gt90"] = (r2s_calc>.90).sum() / total_nnz_neurons
    # stats["sim_gt90"] = total_sim_gt9 / total_nnz_neurons
    # stats["sim_gt99"] = total_sim_gt99 / total_nnz_neurons
    # stats["perc_cos_also_r2_gt90"] =  total_sim_r2_gt9 / total_sim_gt9
    # stats["perc_cos_also_r2_gt99"] =  total_sim_r2_gt99 / total_sim_gt99
    stats["perc_r2_lt0"] =  total_lt_0 / total_nnz_neurons
    
    stats["mean_contr"] =  total_num_contributors / total_neurons
    stats["mean_contr_gt99"] =  total_num_contributors_gt99 / total_r2_gt99 if total_r2_gt99 > 0 else 0
    stats["perc_A_gt01"] =  total_num_contributors / total_A_numel
    stats["perc_A_gt1"] =  total_A_gt1 / total_A_numel

    return  stats



class LinearSpanRegularizer(nn.Module):
    def __init__(self, intermediate_size, num_layers, sparse_args, device):
        super().__init__()
        
        self.intermediate_size = intermediate_size
        self.As =  nn.ParameterList([nn.Parameter(torch.zeros(self.intermediate_size, self.intermediate_size, device=device)) for n in range(num_layers)])

        self.r2s = defaultdict(float)
        self.sparse_args = sparse_args

    def get_A_reg_l1(self):
        reg = 0
        numel = 0
        for A in self.As:
            reg += A.abs().sum()
            numel += A.numel()
        return reg / numel

    def forward(self, hidden_outputs, total_nnz_neurons, print_stats):
        assert len(hidden_outputs) == len(self.As)

        total_norm = 0
        r2s = []
        cnt = 0
        total_sim_gt9 = 0
        total_sim_r2_gt9 = 0
        total_sim_gt99 = 0
        total_sim_r2_gt99 = 0
        total_lt_0 = 0

        total_num_contributors = 0
        total_neurons = 0
        total_A_numel = 0
        total_A_gt1 = 0

        total_num_contributors_gt99 = 0
        total_r2_gt99 = 0

        for n, output in enumerate(hidden_outputs):
            num = str(n)
            # total_nnz_neurons += threshold * self.intermediate_size

            A = self.As[n] * (1 - torch.eye(self.intermediate_size, device= self.As[n].device))

            h = output.clone().detach().view(-1, output.shape[-1])

            s = (h @ A - h)

            # total_norm += s.norm()
            total_norm += (s ** 2 ).mean()

            s_mean = (s.detach().clone()**2).mean(dim=0)
            h_mean = (h.detach().clone()**2).mean(dim=0)

            if self.sparse_args.running_r2_mult != 1:
                r2 = 1 - (s_mean / h_mean ).flatten()
                if not ((num + "running_r2") in  self.r2s.keys()): 
                    self.r2s[num + "running_r2"] =  self.sparse_args.running_r2_mult * r2
                else:
                    self.r2s[num + "running_r2"] = (1-self.sparse_args.running_r2_mult)*self.r2s[num + "running_r2"] + self.sparse_args.running_r2_mult * r2
                span_r2_per_neuron = self.r2s[num + "running_r2"]
            else:
                span_r2_per_neuron = 1 - (s_mean / h_mean  ).flatten()
            
            span_r2_per_neuron = torch.nan_to_num(span_r2_per_neuron, posinf=0, neginf=0)
            total_lt_0 += (span_r2_per_neuron < 0).sum()
            span_r2_per_neuron[span_r2_per_neuron < 0] = 0 # REMOVE LESS THAN 0 
            # r2 = span_r2_per_neuron * mask if mask is not None else span_r2_per_neuron
            r2 = span_r2_per_neuron 
            r2s.append(r2.detach().clone())

            

            total_num_contributors += (A.abs() > .01).sum()
            total_neurons += int(A.shape[0])
            total_A_numel += A.numel()
            total_A_gt1 += (A.abs() > .01).sum()

            r2_gt99 = r2>.99
            total_num_contributors_gt99 += (A.abs() > .01).sum(axis=0)[r2_gt99].sum()
            total_r2_gt99 += r2_gt99.sum()



            sim = h.T @ h 
            norm = h.norm(dim=0) .view(-1, 1)
            sim = sim /  (  norm * norm.T + 1e-6   )
            idx = range(0, sim.shape[1])
            sim[idx, idx] = 0 # so they get ignored
            
            # cos_gt9 = ((sim > .9).sum(axis=1) > 0).to(torch.int) * mask if mask is not None else span_r2_per_neuron
            cos_gt9 = ((sim > .9).sum(axis=1) > 0).to(torch.int) 
            cos_gt99 = ((sim > .99).sum(axis=1) > 0).to(torch.int) 

            total_sim_gt9 += cos_gt9.sum()
            total_sim_r2_gt9 += cos_gt9[r2>.90].sum()
            total_sim_gt99 += cos_gt99.sum()
            total_sim_r2_gt99 += cos_gt99[r2>.99].sum()

            if print_stats:
                nz = r2[r2!=0]
                print("span reg", n, "mean", r2.mean(), "max", r2.max(), "nzmin", nz.min() if nz.numel() > 0 else None, "var", r2.var())
                print("span reg", n, ">.99", (r2>.99).sum(), ">.95", (r2>.95).sum(), ">.90", (r2>.90).sum())
                print("cos_gt9", cos_gt9.sum(), "sim_r2_gt9", cos_gt9[r2>.90].sum(), ">.90", (r2>.90).sum() )
                print("cos_gt99", cos_gt99.sum(), "sim_r2_gt99", cos_gt9[r2>.99].sum(), ">.99", (r2>.99).sum() )
            cnt += 1

        r2s_calc = torch.cat(r2s)

        stats = {}
        stats["r2s"] = r2s
        stats["mean"] = r2s_calc.sum() / total_nnz_neurons
        stats["gt99"] = (r2s_calc>.99).sum() / total_nnz_neurons
        # stats["gt95"] = (r2s_calc>.95).sum() / total_nnz_neurons
        stats["gt90"] = (r2s_calc>.90).sum() / total_nnz_neurons
        stats["sim_gt90"] = total_sim_gt9 / total_nnz_neurons
        stats["sim_gt99"] = total_sim_gt99 / total_nnz_neurons
        stats["perc_cos_also_r2_gt90"] =  total_sim_r2_gt9 / total_sim_gt9
        stats["perc_cos_also_r2_gt99"] =  total_sim_r2_gt99 / total_sim_gt99
        stats["perc_r2_lt0"] =  total_lt_0 / total_nnz_neurons
        
        stats["mean_contr"] =  total_num_contributors / total_neurons
        stats["mean_contr_gt99"] =  total_num_contributors_gt99 / total_r2_gt99  if total_r2_gt99 > 0 else 0
        stats["perc_A_gt01"] =  total_num_contributors / total_A_numel
        stats["perc_A_gt1"] =  total_A_gt1 / total_A_numel

        return total_norm, stats




class PruningTrainer(SparseTrainer, Trainer):
    def __init__(self, sparse_args, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        SparseTrainer.__init__(self, sparse_args)

        self.cos_sims = defaultdict(float)
        self.span_regs = defaultdict(float)

        self.use_span_reg = False
        if self.sparse_args.span_reg_lamb > 0:
            # init_span_reg(self.model)
            cnt = 0
            hidden_size = 0
            for n, mod in self.model.named_modules():
                if isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP):  # this is absolutely terrible, not sure how else to do it for now
                    cnt +=1 
                    hidden_size = mod.c_fc.weight.shape[0]
            assert hidden_size > 0
            self.enable_span_reg(hidden_size, cnt, torch.cuda.current_device())

    def enable_span_reg(self, hidden_size, num_layers, device):
        self.use_span_reg = True
        self.linear_span_regularizer = LinearSpanRegularizer(hidden_size, num_layers, self.sparse_args, device)
        
    def create_optimizer(self, model):
        args = self.args
        span_reg_params = [] if not self.use_span_reg else self.linear_span_regularizer.parameters()
        # span_reg_params = [] 
        optimizer_grouped_parameters = self.patch_coordinator.create_optimizer_groups(model, self.args, self.sparse_args, span_reg_params=span_reg_params)

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        return optimizer

    def training_step(self, model, inputs) -> torch.Tensor:
        self.schedule_threshold(True)
        self.log_prefix = ""
        model.train()
        inputs = self._prepare_inputs(inputs)

        # if is_sagemaker_mp_enabled():
        #     scaler = self.scaler if self.do_grad_scaling else None
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)

        # with self.autocast_smart_context_manager():
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        # if self.do_grad_scaling:
        #     self.scaler.scale(loss).backward()
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
        #     loss = self.deepspeed.backward(loss)
        # else:

        # loss.requires_grad = True  # this is necessary for linear model chooser?
        loss.backward()
        
        if self.sparse_args.span_reg_lamb > 0: # DOES NOT WORK WITH GRAD ACCUMULATION - HAD  TO DO THIS FOR SPAN REG
            torch.nn.utils.clip_grad_norm_(self.linear_span_regularizer.parameters(), self.args.max_grad_norm)


        print_stats = False
        if self.sparse_args.track_eval_cos or self.sparse_args.adjust_grad_lamb != 0:
            update_running_cos(self.cos_sims, self.sparse_args, model, outputs) # always update this for logging
            print_stats = (self.state.global_step % (self.args.logging_steps*4)) == 0   # only print stats twice per epoch
            
            prune_threshold = self.patch_coordinator.patcher_context.get_context_data("threshold")

            all_sim, all_neuron_scores, all_neuron_multipliers = adjust_grad(self.cos_sims, self.sparse_args, prune_threshold, model, outputs, self.sparse_args.adjust_grad_lamb, print_stats)


        if self.sparse_args.dense_pruning_method == "sage":
            update_sage(model, self.state.global_step, self.sparse_args)

        if self.sparse_args.zero_pruned and self.state.global_step > self.args.warmup_steps:
            self.zero_pruned_weights(model)

        return loss.detach()

    # @torch.no_grad()
    # def zero_pruned_weights(self, model):
    #     for module in model.modules():
    #         if isinstance(module, MaskedLinear):
    #             masked_weights, bias =  module.get_masked_weights_bias()
    #             module.weight[:] = masked_weights.clone().detach()


        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        We override the default loss in SparseTrainer because it throws an 
        error when run without distillation
        """
        # print(3, "torch alloc", torch.cuda.memory_allocated(0), "torch alloc 1", torch.cuda.memory_allocated(1), "torch reserved", torch.cuda.memory_reserved(0))

        
        if "global_topK" in self.sparse_args.dense_pruning_method:
            threshold = self.patch_coordinator.patcher_context.get_context_data("threshold")
            distribute_thresholds(self.sparse_args, model, threshold)
        
        outputs = model(**inputs)
        # outputs["past_key_values"] = None
        # outputs["hidden_states"] = None
        # outputs["attentions"] = None
        # torch.cuda.empty_cache()
        

        if self.sparse_args.track_eval_cos or self.sparse_args.adjust_grad_lamb != 0:
            update_current_total_cos(self.cos_sims, self.sparse_args, model, outputs) 

        labels = inputs["labels"]
        logits = outputs["logits"]
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()
        logits = torch.argmax(logits, axis=-1)

        # real = wikisql_validation.tokenizer.decode(labels[0])
        # pred = wikisql_validation.tokenizer.decode(logits[0])
        # pred = wikisql_validation.tokenizer.decode(inputs["input_ids"][0][100:140])
        # print(real, pred, acc_mask, inputs["label_mask"])

        acc = None
        if "label_mask" in inputs:
            mask = inputs["label_mask"]
            correct_labels = ((logits[:] == labels[:])*mask).sum() / (mask.sum())  # full acc
            acc = ((logits[:] == labels[:])*mask).sum(axis=1, keepdims=True)
            acc = (acc == mask.sum(axis=1, keepdims=True)).sum() / labels.shape[0]
            self.metrics["accuracy"] += acc
            self.metrics["correct_labels"] += correct_labels

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if outputs["raw_loss"] is not None:
            if "target_mask" in inputs.keys():
                print("Found target mask")
                self.metrics["raw_loss"] +=  (outputs["raw_loss"]* inputs["target_mask"]).mean() 
                self.metrics["ppl"] +=  torch.exp((outputs["raw_loss"]* inputs["target_mask"]).mean() )
            else:
                self.metrics["raw_loss"] +=  outputs["raw_loss"].mean() 
                self.metrics["ppl"] +=  torch.exp(outputs["raw_loss"].mean())
                
        # print(4, "torch alloc", torch.cuda.memory_allocated(0), "torch alloc 1", torch.cuda.memory_allocated(1), "torch reserved", torch.cuda.memory_reserved(0))


        loss, distil_loss = self.patch_coordinator.distil_loss_combine(loss, inputs, outputs)
        loss = loss.mean()
        
        if self.sparse_args.span_reg_lamb > 0:
            print_stats = (self.state.global_step % (self.args.logging_steps*4)) == 0   # only print stats twice per epoch
            # r2stats = udpate_span_regs(self.span_regs, self.sparse_args, model, outputs, print_stats)
            # span_reg = 0
            # for reg in outputs["all_span_reg"]:
            #     span_reg += reg.norm()

            span_reg = sum(outputs["all_span_reg"]).mean()

            
            total_nnz_neurons  = 0
            for n, mod in model.named_modules():
                if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
                if isinstance(mod.c_fc, MaskedLinear):
                    masked_weights, bias, threshold, mask = mod.c_fc.get_masked_weights_bias() 
                    total_nnz_neurons += mask[:, 0].sum()
                else:
                    total_nnz_neurons += mod.c_fc.weight.shape[0]
            span_reg, r2stats = self.linear_span_regularizer(outputs["all_neuron_outputs_fc"], total_nnz_neurons, print_stats)

            if self.sparse_args.mask_span_reg_lamb > 0:
                do_mask_span_reg_lamb(r2stats["r2s"], self.sparse_args, model, self.sparse_args.mask_span_reg_lamb, print_stats)

            # A_reg = span_reg_A(model)

            # r2stats = get_r2_stats(model, outputs, total_nnz_neurons)

            self.metrics["span_reg_prelamb"] += float(span_reg)

            self.metrics["span_reg_r2"] += r2stats["mean"]
            self.metrics["span_reg_r2_gt99"] += r2stats["gt99"]
            self.metrics["span_reg_r2_gt90"] += r2stats["gt90"]
            self.metrics["cos_sim_gt90"] += r2stats["sim_gt90"]
            self.metrics["perc_cos_also_r2_gt90"] += r2stats["perc_cos_also_r2_gt90"]
            self.metrics["perc_cos_also_r2_gt99"] += r2stats["perc_cos_also_r2_gt99"]
            self.metrics["sim_gt90"] += r2stats["sim_gt90"]
            self.metrics["sim_gt99"] += r2stats["sim_gt99"]
            self.metrics["perc_r2_lt0"] += r2stats["perc_r2_lt0"]
            self.metrics["mean_contr"] += r2stats["mean_contr"]
            self.metrics["mean_contr_gt99"] += r2stats["mean_contr_gt99"]
            self.metrics["perc_A_gt01"] += r2stats["perc_A_gt01"]
            self.metrics["perc_A_gt1"] += r2stats["perc_A_gt1"]
            
            if self.sparse_args.A_reg_lamb > 0:
                A_reg = self.linear_span_regularizer.get_A_reg_l1()
                # A_reg = get_A_reg_l1(self.model)
                self.metrics["A_reg_prelamb"] += float(A_reg)
                loss += self.sparse_args.span_reg_lamb * span_reg  + self.sparse_args.A_reg_lamb *A_reg
            else:
                loss += self.sparse_args.span_reg_lamb * span_reg

        regu_loss, lamb, weight_lamb, info = self.patch_coordinator.regularization_loss(model)
        for kind, values in info.items():
            if kind == "total":
                suffix = ""
            else:
                suffix = "_" + kind

            for k, v in values.items():
                self.metrics[k + suffix] += float(v)
        self.metrics["ce_loss"] += float(loss)
        self.metrics["distil_loss"] += float(distil_loss)
        self.loss_counter += 1

        loss = loss + regu_loss * lamb 
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            # if is_sagemaker_mp_enabled():
            #     raw_outputs = smp_forward_only(model, inputs)
            #     if has_labels:
            #         if isinstance(raw_outputs, dict):
            #             loss_mb = raw_outputs["loss"]
            #             logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
            #         else:
            #             loss_mb = raw_outputs[0]
            #             logits_mb = raw_outputs[1:]

            #         loss = loss_mb.reduce_mean().detach().cpu()
            #         logits = smp_nested_concat(logits_mb)
            #     else:
            #         loss = None
            #         if isinstance(raw_outputs, dict):
            #             logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
            #         else:
            #             logits_mb = raw_outputs
            #         logits = smp_nested_concat(logits_mb)
            # else:
                if has_labels:
                    # with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    # with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        # outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
    
    def _save(self, output_dir = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        do_prune = self.sparse_args.dense_pruning_method != "disabled" or self.sparse_args.attention_pruning_method  != "disabled"
        if do_prune:
            print("Compiling model")
            model_copy = copy.deepcopy(self.model)
            self.patch_coordinator.compile_model(model_copy)
            compiled_dir = os.path.join(output_dir, "compiled")
            print(f"Saving compiled model checkpoint to {compiled_dir}")
            model_copy.save_pretrained(compiled_dir, state_dict=state_dict)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        # torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))




@torch.no_grad()
def update_sage(model, global_step, sparse_args):

    all_ipt = []
    cnt = 0
    for n, module in enumerate(model.modules()):
        if not (isinstance(module, GPTNeoMLP) or isinstance(module, GPT2MLP)): continue
        # again this is very gross, formalize if this works

        all_ipt.append(module.c_fc.weight * module.c_fc.weight.grad)
        all_ipt.append(module.c_proj.weight * module.c_proj.weight.grad)


    cnt = 0
    for n, module in enumerate(model.modules()):
        if not isinstance(module, SageLinearPruningContextModule): continue
        local_step = global_step % sparse_args.sage_delta_T
        update_step = global_step // sparse_args.sage_delta_T
        if local_step == 0: 
            module.exp_avg_ipt[:] = sparse_args.sage_beta_3 * module.exp_avg_ipt + (1 - sparse_args.sage_beta_3) * module.ipt

            if sparse_args.sage_beta_meta > 0 and sparse_args.sage_beta_meta < 1:
                module.exp_avg_unc[:] = sparse_args.sage_beta_meta * module.exp_avg_unc + (1 - sparse_args.sage_beta_meta) * (module.ipt-module.exp_avg_ipt).abs()

            elif sparse_args.sage_beta_meta == 2.:
                module.exp_avg_unc[:] = (update_step * module.exp_avg_unc + (module.ipt-module.exp_avg_ipt)**2 )/(update_step+1)
            
            module.ipt[:] = (all_ipt[cnt]).abs().detach()
        else:
            module.ipt[:] = (module.ipt * local_step + (all_ipt[cnt]).abs().detach())/(local_step+1)

        cnt += 1


