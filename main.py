# from tkinter import N
# from numpy.core.fromnumeric import size
import collections 
import collections.abc
from http.client import GATEWAY_TIMEOUT
from nn_pruning.modules.masked_nn import MaskedLinear
import torch
import datasets
import transformers
from transformers import TrainerCallback
import os
import argparse
from nn_pruning.sparse_trainer import SparseTrainer, TimingModule
from datasets import load_dataset
from datasets import load_metric
from data import get_dataset, get_dataset_e2e_nlg, get_datasets_wikitext, get_dataset_wikisql_distil, get_datasets_samsum
from transformers import TrainingArguments
import torch 
# from transformers import AutoModelForCausalLM, AutoConfig
# from transformers import AutoConfig
from nn_pruning.patch_coordinator import ModelPatchingCoordinator
from nn_pruning.inference_model_patcher import optimize_model
from model import GPTNeoForCausalLM, GPT2LMHeadModel, GPTNeoMLP, GPT2MLP
import numpy as np
from torch import nn
import pandas as pd
from utils import PruningTrainer, args_to_hparams, init_span_reg
import os
from beam_decode import eval_write, eval_rouge
from transformers import AutoTokenizer
import pickle 
from transformers.modeling_utils import (
    Conv1D,
)

import random

from transformers import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
os.environ["WANDB_DISABLED"] = "true"
DISABLE_MLFLOW_INTEGRATION = True

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='PyTorch GPT-Neo ft script')


parser.add_argument('--a_leftover', default=0, type=float, help='amount of params left over after pruning')
parser.add_argument('--a_distil', action='store_true', help='net was distilled')
parser.add_argument('--a_method', default="", help='method of pruning used')
parser.add_argument('--unmodify_gpt2', action='store_true', help='use gpt2 with conv1d')
parser.add_argument('--a_validation', action='store_true', help='use validation for analysis')



parser.add_argument('--train', action='store_true', help='train the net')
parser.add_argument('--analyze', action='store_true', help='analyze the net by gathering sensitivity and uniqueness')
parser.add_argument('--eval', action='store_true', help='eval the net')
parser.add_argument('--eval_beam', action='store_true', help='evaluate the net with BEAM on the test set')
parser.add_argument('--eval_rouge', action='store_true', help='evaluate the net with rouge metric on test')
parser.add_argument('--eval_beam_speed', action='store_true', help='evaluate the speed of the net with BEAM on the test set')
parser.add_argument('--eval_output_dir', default="./", help='location of saved beam output')
parser.add_argument('--save_outputs', action='store_true', help='save net outputs (for later distillation)')
parser.add_argument('--eval_slide', action='store_true', help='evaluate the net with sliding window ppl on the test set')

parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--quiet', action='store_true', help='no prints')

parser.add_argument('--task', default="wikisql", help='which task to run', choices=('wikisql', 'e2e_nlg', 'wikitext', 'wikisql_distil', 'samsum'))
parser.add_argument('--dataset_path', default="/home/azureuser/FT_wikisql_v8/", help='location of data corpus')
parser.add_argument('--tokenizer_path', required=True,  help='location of tokenizer')



parser.add_argument('--model_type', required=True, help='type of model', choices=('gpt-neo', 'gpt-2'))
parser.add_argument('--model_path', required=True, help='location of model')
parser.add_argument('--state_dict_path', default=None, help='location of model')
parser.add_argument('--output_dir', default=None, help='location of output dir')
parser.add_argument('--save_model', action='store_true', help='save the net')


parser.add_argument('--prune_then_train', action='store_true', help='prune then train the net')
parser.add_argument('--sensitivity_preprune', action='store_true', help='pre prune with sensitivity')
parser.add_argument('--sensitivity_preprune_beta', default=2, type=float, help='pre prune beta')



parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
parser.add_argument('--mask_lr', default=0.01, type=float, help='mask scores learning rate')
parser.add_argument('--wd', default=.1, type=float, help='weight decay')
parser.add_argument('--regu_lamb', default=2, type=float, help='regu lambda')
parser.add_argument('--weight_regu_lamb', default=.05, type=float, help='weight regu lambda')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max grad norm')
parser.add_argument('--label_smoothing', default=0, type=float, help='label smoothing')
parser.add_argument('--prune_leftover', default=.1, type=float, help='amount of params left over after pruning')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--epochs', default=5, type=int, help='epochs')
parser.add_argument('--schedule', default="linear", help='schedule type', choices=('linear', 'cos', 'constant'))
parser.add_argument('--token_max_len', default=512, type=int, help='token max len')
parser.add_argument('--adam_beta1', default=.9, type=float, help='learning rate')
parser.add_argument('--adam_beta2', default=.999, type=float, help='learning rate')
parser.add_argument('--adam_epsilon', default=1e-4, type=float, help='adam epsilon')
parser.add_argument('--zero_pruned', action='store_true', help='zero out pruned weights')
parser.add_argument('--warmup_percent', default=.1, type=float, help='warmup percent')
parser.add_argument('--initial_warmup', default=1, type=int, help='initial_warmup')

parser.add_argument('--mask_init',  default="constant", help='mask init', choices=('constant', 'uniform', 'kaiming'))
parser.add_argument('--mask_init_scale', default=0, type=float, help='mask init')
parser.add_argument('--mask_frozen', action='store_true', help='freeze mask and leave untrained')

parser.add_argument('--distil_teacher_name_or_path', default=None, type=str, help='distil_teacher_name_or_path')
parser.add_argument('--distil_alpha_ce', default=0.5, type=float, help='distil_alpha_ce')
parser.add_argument('--distil_alpha_teacher', default=0.5, type=float, help='distil_alpha_teacher')
parser.add_argument('--distil_temperature', default=2.0, type=float, help='distil_temperature')

parser.add_argument('--scale_pruned', action='store_true', help='use scale_pruned')
parser.add_argument('--scale_fc', action='store_true', help='use scale_fc')
parser.add_argument('--scale_proj', action='store_true', help='use scale_proj')
parser.add_argument('--scale_params_learning_rate', default=1e-4, type=float, help='scale_params_learning_rate')


parser.add_argument('--soft_temperature', default=0, type=float, help='regu soft_temperature')

parser.add_argument('--adjust_grad_lamb', default=0, type=float, help='adjust grad lamb')
parser.add_argument('--anti_gum', action='store_true', help='use anti gum')
parser.add_argument('--running_cos_mult', default=.999, type=float, help='running cos multiplier')
parser.add_argument('--running_cos_method', default="decaying", help='running cos method', choices=('decaying', 'exp_avg', "sage"))
parser.add_argument('--track_eval_cos', action='store_true', help='track eval cos')
parser.add_argument('--uniqueness_reg_mask', action='store_true', help='uniqueness_reg_mask')
parser.add_argument('--adjust_grad_do_mult', action='store_true', help='adjust_grad_do_mult')
parser.add_argument('--cpu_cos_sim',  action='store_true', help='cos sim stored on cpu')

parser.add_argument('--adjust_mask_grad', action='store_true', help='adjust_mask_grad')



parser.add_argument('--sage_delta_T', default=5, type=int, help='sage delta T')
parser.add_argument('--sage_beta_3', default=.85, type=float, help='sage beta 3')
parser.add_argument('--sage_beta_meta', default=1, type=float, help='sage beta meta')

parser.add_argument('--train_only_bias_ln', action='store_true', help='train only bias and layernorm, no weights')

parser.add_argument('--dense_pruning_method', default="disabled", help='dense pruning method', choices=('disabled', 'topK', 'magnitude', 'threshold', 'sigmoied_threshold',  "l0", "sage", "extra_soft", "global_topK", "group_magnitude"))
parser.add_argument('--dense_pruning_submethod', default="default", help='dense pruning submethod', choices=('default', '1d', '1d_alt', '1d_alt_plus', '1d_only'))
parser.add_argument('--attention_pruning_method', default="disabled", help='attention pruning method', choices=('disabled', 'topK', 'magnitude', 'threshold', 'sigmoied_threshold'))
parser.add_argument('--regularization', default="disabled", help='regularization method', choices=('disabled', 'l0', 'l1'))
parser.add_argument('--weight_regularization', default="disabled", help='regularization method', choices=('disabled', "uniqueness"))


parser.add_argument('--train_samples', default=None, type=int, help='number of training samples to use')
parser.add_argument('--valid_samples', default=None, type=int, help='number of validation samples to use')

parser.add_argument('--adjust_mask_uniqueness', action='store_true', help='adjust mask uniqueness')


parser.add_argument('--span_reg_lamb', default=0, type=float, help='span_reg_lamb')
parser.add_argument('--mask_span_reg_lamb', default=0, type=float, help='adjust mask scores with lamb*span_reg_lamb')
parser.add_argument('--span_reg_A_learning_rate', default=1e-2, type=float, help='span_reg_A_learning_rate')
parser.add_argument('--A_reg_lamb', default=0, type=float, help='span_reg_lamb')
parser.add_argument('--running_r2_mult', default=1, type=float, help='running_r2_mult')



parser.add_argument('--opt_span_reg_only', action='store_true', help='opt_span_reg_only')


if __name__ == "__main__": 
    args = parser.parse_args()
    parser_args = parser.parse_args()

    if args.unmodify_gpt2:
        GPT2MLP.USE_CONV1D_GPT2 = True


    if args.prune_leftover == 1:
        # print("prune leftover is 1 - disabling pruning")
        # args.dense_pruning_method = "disabled"
        # args.dense_pruning_submethod = "default"
        # args.regularization = "disabled"

        
        print("prune leftover is 1!!")

    print("arguments")
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    set_seed(args.seed)

    random.seed(args.seed)

    if args.dense_pruning_method == "extra_soft":
        assert args.soft_temperature > 0
        print("doing extra_soft pruning")
    else:
        assert args.soft_temperature == 0

    # if args.adjust_grad_lamb != 0:
    #     print("Adjusting grad with lambda", args.adjust_grad_lamb)
    #     print("tracking eval cos automatically")
    #     args.track_eval_cos = True

    if args.dense_pruning_method == "sage":
        print("doing sage pruning - using default submethod")
        0/0 # not supported -need to fix or remove
        dense_pruning_submethod = "default"

    datasets.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    print(f"Using transformers v{transformers.__version__} and datasets v{datasets.__version__} and torch v{torch.__version__}")

    model_name = args.model_path

    # print(1, "torch alloc", torch.cuda.memory_allocated(0), "torch alloc 1", torch.cuda.memory_allocated(1), "torch reserved", torch.cuda.memory_reserved(0))

    if args.task == "wikisql":
        data_train = get_dataset(args.tokenizer_path, os.path.join(args.dataset_path, "train.jsonl"), "train", args.train_samples, args.token_max_len, args.token_max_len, False, lower=False)
        data_validation = get_dataset(args.tokenizer_path,os.path.join(args.dataset_path, "validation.jsonl"), "validation", args.valid_samples, args.token_max_len, args.token_max_len, False, lower=False)
        data_test = get_dataset(args.tokenizer_path, os.path.join(args.dataset_path, "test.jsonl"), "test", args.valid_samples, args.token_max_len, args.token_max_len, False, lower=False)
    elif args.task == "e2e_nlg":
        data_train, data_validation, data_test = load_dataset(args.dataset_path, split=['train', 'validation', 'test'])

        data_train = get_dataset_e2e_nlg(args.tokenizer_path, data_train, "train", args.train_samples, args.token_max_len, args.token_max_len, args.seed, False, lower=False)
        data_validation = get_dataset_e2e_nlg(args.tokenizer_path,data_validation, "validation", args.valid_samples, args.token_max_len, args.token_max_len, args.seed, False, lower=False)
        data_test = get_dataset_e2e_nlg(args.tokenizer_path, data_test, "test", args.valid_samples, args.token_max_len, args.token_max_len, args.seed, False, lower=False)
        
    elif args.task == "wikitext":
        data_train, data_validation, data_test = load_dataset(args.dataset_path, name='wikitext-103-v1', split=['train', 'validation', 'test'])

        data_train = get_datasets_wikitext(args.tokenizer_path, data_train, args.token_max_len, args.train_samples, None, seed=args.seed, is_analyze=args.analyze)
        data_validation = get_datasets_wikitext(args.tokenizer_path, data_validation, args.token_max_len,  args.valid_samples, None, seed=args.seed, is_analyze=args.analyze)


    elif args.task == "wikisql_distil":
        with (open(os.path.join(args.dataset_path, "train_teacher_outputs.pkl"), "rb")) as openfile:
            dataset_train = pickle.load(openfile)

        with (open(os.path.join(args.dataset_path, "val_teacher_outputs.pkl"), "rb")) as openfile:
            dataset_validation = pickle.load(openfile)
        
        
        data_train = get_dataset_wikisql_distil(args.tokenizer_path, dataset_train, "train", args.train_samples, args.token_max_len, args.token_max_len, False, lower=False)
        data_validation = get_dataset_wikisql_distil(args.tokenizer_path, dataset_validation, "validation", args.valid_samples, args.token_max_len, args.token_max_len, False, lower=False)

    elif args.task == "samsum":
        data_train, data_validation, data_test = load_dataset(args.dataset_path, name='samsum', split=['train', 'validation', 'test'])

        data_train =  get_datasets_samsum(args.tokenizer_path, data_train, args.token_max_len, args.train_samples)
        data_validation =  get_datasets_samsum(args.tokenizer_path, data_validation, args.token_max_len, args.valid_samples)
        data_test =  get_datasets_samsum(args.tokenizer_path, data_test, args.token_max_len, args.valid_samples, is_test=True)
        


    else:
        print("not implemented")
        0/0


    learning_rate = args.lr
    n_gpu = torch.cuda.device_count()
    batch_size = args.batch_size
    epoch_steps = len(data_train) // (batch_size*n_gpu)
    num_train_epochs = args.epochs 
    train_steps = int(epoch_steps * num_train_epochs)
     
    if epoch_steps > 8:
        logging_steps = int(epoch_steps / 8)
    else:
        logging_steps = int(epoch_steps) # when debugging
    # warmup_steps = int(train_steps * 0.005) 
    warmup_steps = int(train_steps * args.warmup_percent) 
    eval_steps = int(epoch_steps)   # eval every  epoch

    # save_steps = epoch_steps
    save_steps = int((epoch_steps * args.epochs ))  # save once at end (should change this to best perf but this is fine)


    print("eval steps", eval_steps)
    print("batch_size", batch_size)
    print("epoch_steps", epoch_steps)
    print("n_gpu", n_gpu)

    save_strategy = "no"
    if args.save_model:
        save_strategy = "steps"
    if args.output_dir is None:
        output_dir = "checkpoints"
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        output_dir = os.path.join(args.output_dir, "checkpoints")

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps= eval_steps,
        save_strategy=save_strategy,
        save_steps = save_steps,
        # gradient_accumulation_steps=1,
        eval_accumulation_steps=1,
        max_steps=train_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=args.wd,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        disable_tqdm=True,
        report_to=None,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        remove_unused_columns=not args.analyze
    )
    print(training_args)


    sparse_args = args_to_hparams(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_constructor = GPTNeoForCausalLM if args.model_type == "gpt-neo" else GPT2LMHeadModel

    mpc = ModelPatchingCoordinator(
        sparse_args=sparse_args, 
        device=device, 
        cache_dir="checkpoints", 
        model_name_or_path=model_name,
        logit_names=["logits"], 
        teacher_constructor=teacher_constructor)

    if args.model_type == "gpt-neo":
        train_model = GPTNeoForCausalLM.from_pretrained(model_name, output_hidden_states=args.span_reg_lamb > 0).to(device)
    else:
        train_model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=args.span_reg_lamb > 0).to(device)


    log_df = []

    class LogDfCallback(TrainerCallback):
        def on_evaluate(self, my_args, state, control, metrics=None, **kwargs):
            if state.is_local_process_zero:
                logs = {**metrics, **vars(parser_args)}

                for n, mod in train_model.named_modules():
                    if not isinstance(mod, MaskedLinear): continue
                    if not mod.is_scaling: continue
                    if args.scale_fc:
                        logs[n + "_scale_mean"] = mod.out_scale.mean()
                        logs[n + "_scale_var"] = mod.out_scale.var()

                    info = mod.get_sparsity_info()

                    logs[n + "_numel"] = info["numel"]
                    logs[n + "_nnz"] = info["nnz"]

                log_df.append(logs)
    
    label_smoothing = args.label_smoothing
    if args.distil_teacher_name_or_path is not None:
        print("Using distillation - turning off label smoothing!")
        label_smoothing = 0
    
    print("label smoothing", label_smoothing)
    train_model.label_smoothing = label_smoothing

    if args.span_reg_lamb > 0:
        print("using span_reg_lamb", args.span_reg_lamb)
        # init_span_reg(train_model)
        print("TRYING NEW SPAN REG METHOD in Optimizer!")

    if args.adjust_grad_lamb != 0:
        print("tracking train cos automatically")
        for n, mod in enumerate(train_model.modules()):
            if not (isinstance(mod, GPTNeoMLP) or isinstance(mod, GPT2MLP)): continue
            mod.cos_sim = True



    if args.sensitivity_preprune:
        print("doing pre-train sensitivity pruning")
        beta = args.sensitivity_preprune_beta

        layer_sensitivity = {}
        for n, mod in train_model.named_modules():
            if not isinstance(mod, GPTNeoMLP): continue
            layer_sensitivity[n] = mod.c_fc.weight.new_zeros(( mod.c_fc.weight.shape[0], ))

        correct = 0
        print("collecting outputs")
        samples = 0
        for idx, inputs in enumerate(data_train):
            samples += 1
            # if idx > 2000: print("ending at 2000"); break
            train_model.zero_grad()
            input_ids = inputs["input_ids"].to(train_model.device)
            labels = inputs["labels"].to(train_model.device)
            mask = inputs["label_mask"].to(train_model.device)
            outputs = train_model(input_ids=input_ids, labels=labels, label_mask=mask )
        
            logits = outputs["logits"].to(train_model.device)
            logits = logits[..., :-1, :].contiguous().to(train_model.device)
            labels = labels[..., 1:].contiguous().to(train_model.device)
            logits = torch.argmax(logits, axis=-1)
            acc = ((logits[:] == labels[:])*mask).sum() == mask.sum()
            correct += acc
            if idx % 10 == 0:
                print(idx, "running acc", correct / (idx + 1))

            loss = outputs["loss"].mean()
            loss.backward()

            with torch.no_grad():
                seen = []
                for n, mod in train_model.named_modules():
                    if not isinstance(mod, GPTNeoMLP): continue
                    sensitivity = (mod.c_fc.weight * mod.c_fc.weight.grad).sum(dim=1)
                    if mod.c_fc.bias is not None:
                        sensitivity += mod.c_fc.bias * mod.c_fc.bias.grad
                    sensitivity += (mod.c_proj.weight * mod.c_proj.weight.grad).sum(dim=0)
                    sensitivity = torch.abs(sensitivity)
                    sensitivity[torch.isnan(sensitivity)] = 0.0
                    sensitivity[torch.isinf(sensitivity)] = 0.0

                    # layer_sensitivity[n] = beta*layer_sensitivity[n] + (1-beta)*sensitivity
                    layer_sensitivity[n] = layer_sensitivity[n] + sensitivity
         
        for n, mod in train_model.named_modules(): # mean over dataset instead of running mean
            if not isinstance(mod, GPTNeoMLP): continue
            layer_sensitivity[n] = layer_sensitivity[n] / samples
        
        with torch.no_grad():
            print("selecting neurons")
            for n, mod in train_model.named_modules():
                if not isinstance(mod, GPTNeoMLP): continue
                print(n,  layer_sensitivity[n].mean(),  layer_sensitivity[n].max(),  layer_sensitivity[n].min())
                _, idx =  layer_sensitivity[n].clone().flatten().sort(descending=True)
                j = int(args.prune_leftover * layer_sensitivity[n].numel())

                mod.c_fc.weight[idx[j:], :] = 0
                mod.c_fc.bias[idx[j:]] = 0
                mod.c_proj.weight[:, idx[j:]] = 0

                sens_selected = layer_sensitivity[n][:j]
                sens_scale = sens_selected.sum() / layer_sensitivity[n].sum()

                print("sens_scale", sens_scale)

                mod.c_proj.weight[:] = mod.c_proj.weight / sens_scale
        
        print("optimizing model")
        train_model = optimize_model(train_model, "dense", keep_dim_mode="1d_alt")




    if args.train:
        with torch.no_grad():
            # train_model.transformer.wte.weight.data.normal_(mean=0.0, std=0.02)
            embed_shape = train_model.transformer.wte.weight.shape
            decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
            decoder.weight = train_model.transformer.wte.weight  # Tied weights with input
            train_model.set_output_embeddings(decoder)

        print("Patching Model")
        mpc.patch_model(train_model)

    if args.state_dict_path:
        print("Loading state dict after patching...")
        state_dict = torch.load(args.state_dict_path)
        train_model.load_state_dict(state_dict, strict=False)

        
        # for n, mod in train_model.named_modules():
        #     if not isinstance(mod, MaskedLinear): continue
        #     if not mod.is_scaling: continue

        #     print(mod.out_scale)
        #     print(mod.out_scale.mean(), mod.out_scale.min(), mod.out_scale.max())
        
        # 0/0




    if args.prune_then_train:
        mpc.patch_model(train_model)
        print("pruning then training")
        mpc.compile_model(train_model)

        print("optimizing model")

        keep_dim_mode = "default"
        if args.dense_pruning_method == "uniqueness":
            keep_dim_mode = "uniqueness"
        elif args.dense_pruning_submethod == "1d":
            keep_dim_mode = "1d"
        elif args.dense_pruning_submethod == "1d_alt":
            keep_dim_mode = "1d_alt"
        
        train_model = optimize_model(train_model, "dense", keep_dim_mode=keep_dim_mode).to(device)

        # sparse_args.dense_pruning_method = "disabled"
        # sparse_args.dense_pruning_submethod = "default"
        # sparse_args.regularization = "disabled"

        # mpc = ModelPatchingCoordinator(
        #     sparse_args=sparse_args, 
        #     device=device, 
        #     cache_dir="checkpoints", 
        #     model_name_or_path=model_name,
        #     logit_names="logits", 
        #     teacher_constructor=None)
        # mpc.patch_model(train_model)





    if args.train:
        trainer = PruningTrainer(
            sparse_args=sparse_args,
            args=training_args,
            model=train_model,
            train_dataset=data_train,
            eval_dataset=data_validation,
            callbacks=[LogDfCallback]
        )

        trainer.set_patch_coordinator(mpc)

        print("training")
        trainer.train()

        # trainer.is_in_train = True

        print("evaluating")
        results = trainer.evaluate()
        print("results")
        print(results)

        if args.output_dir:
            print("saving results")
            log_file = os.path.join(args.output_dir, 'log.df')
            pd.DataFrame(log_df).to_pickle(log_file)

            # if args.track_eval_cos or args.adjust_grad_lamb != 0:
            #     print("saving self.manual_log_df")
            #     log_file = os.path.join(args.output_dir, 'manual_log_df.df')
            #     pd.DataFrame(trainer.manual_log_df).to_pickle(log_file)
        

        if args.scale_fc or args.scale_proj:
            print("Printing scale results")
            for n, mod in train_model.named_modules():
                if not isinstance(mod, MaskedLinear): continue
                if not mod.is_scaling: continue

                print(n)
                print(mod.out_scale)
                print(mod.out_scale.mean(), mod.out_scale.min(), mod.out_scale.max())
                print()

        if args.span_reg_lamb > 0:
            print("Printing A  results")
            for n, A in enumerate(trainer.linear_span_regularizer.As):

                print(n)
                print(A)
                print(A.abs().mean(), A.min(), A.max(), (A.abs() > .01) / A.numel(), (A.abs() > .1) / A.numel())
                print()

        
        if False:
            prune_df = []

            print("compiling") # compile first so all the mask calculations don't affect timing
            mpc.compile_model(train_model)

            # time_model = TimingModule(train_model)

            trainer = PruningTrainer(
                sparse_args=sparse_args,
                args=training_args,
                model=time_model,
                train_dataset=data_train,
                eval_dataset=data_validation,
            )
            trainer.set_patch_coordinator(mpc)
            print("evaluating validation set ")
            results = trainer.evaluate()
            print("results")
            print(results)
            
            cudaEvalTime, cudaEvalCount = time_model.get_results()
            print("cuda time", cudaEvalTime)

            prune_df.append({"cuda_time": cudaEvalTime, "dataset": "validation", "compressed": False, "num_params":train_model.num_parameters(), **results, **vars(parser_args)})

            time_model.reset()

            print("evaluating pruning")

            print("optimizing model")

            keep_dim_mode = "default"
            if args.dense_pruning_submethod == "1d":
                keep_dim_mode = "1d"
            elif args.dense_pruning_submethod == "1d_alt":
                keep_dim_mode = "1d_alt"
            
            pruned_train_model = optimize_model(train_model, "dense", keep_dim_mode=keep_dim_mode)

            size_diff = pruned_train_model.num_parameters() / train_model.num_parameters()

            print(f"reduced model to {size_diff} of original size")
            
            pruned_time_model = TimingModule(pruned_train_model)
            
            trainer = PruningTrainer(
                sparse_args=sparse_args,
                args=training_args,
                model=pruned_time_model,
                train_dataset=data_train,
                eval_dataset=data_validation,
            )

            trainer.set_patch_coordinator(mpc)

            print("pruned evaluation")

            pruned_results = trainer.evaluate()
            print(pruned_results)

            cudaEvalTime, cudaEvalCount = pruned_time_model.get_results()
            print("cuda time", cudaEvalTime)

            prune_df.append({"cuda_time": cudaEvalTime, "dataset": "validation", "compressed": True, "num_params":pruned_train_model.num_parameters(), **pruned_results, **vars(parser_args)})

            print("done")

            if args.output_dir:
                print("saving prune results")
                prune_log_file = os.path.join(args.output_dir, 'prune_log.df')
                pd.DataFrame(prune_df).to_pickle(prune_log_file)


    if args.eval:
        print("evaluating")
        
        time_model = TimingModule(train_model)

        trainer = PruningTrainer(
            sparse_args=sparse_args,
            args=training_args,
            model=time_model,
            train_dataset=data_train,
            eval_dataset=data_validation,
        )
        trainer.set_patch_coordinator(mpc)


        import time
        print("evaluating validation set ")
        start = time.time()
        results = trainer.evaluate()
        end = time.time()
        print("results")
        print(results)
        
        cudaEvalTime, cudaEvalCount = time_model.get_results()
        print("cuda time", cudaEvalTime)
        print("backup time", end - start)
        print("max memory", torch.cuda.max_memory_allocated() / 1e9)


    if args.eval_slide:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, max_length=None, add_special_tokens=True)
        encodings = tokenizer("\n\n".join(data_test["text"]), return_tensors="pt")
        print("label smoothing off for eval")
        train_model.label_smoothing = 0

        max_length = args.token_max_len
        stride = 1024
        print("Doing eval slide, total len", encodings.input_ids.size(1))

        nlls = []
        for i in range(0, encodings.input_ids.size(1), stride):
            if i % 1000 == 0: print("step", i)

            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            # print(len(encodings),  encodings.input_ids.shape, begin_loc, end_loc)

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -1   # my code uses -1 as ignore index for xent

            with torch.no_grad():
                outputs = train_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

        print("EVAL SLIDE PPL STRIDE", stride)
        print(ppl)
        print(ppl)

    if args.eval_rouge:
        print("evaluating test set with beam search")
        print("label smoothing off for eval")
        train_model.label_smoothing = 0
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, max_length=None, add_special_tokens=True)
        eval_rouge(data_test, tokenizer, train_model, quiet=False)


    if args.eval_beam:
        # eval_df = []

        # trainer = PruningTrainer(
        #     sparse_args=sparse_args,
        #     args=training_args,
        #     model=train_model,
        #     train_dataset=data_train,
        #     eval_dataset=data_validation,
        # )
        # trainer.set_patch_coordinator(mpc)
        # print("evaluating validation set ")
        # results = trainer.evaluate()
        # print("results")
        # print(results)
        
        # trainer = PruningTrainer(
        #     sparse_args=sparse_args,
        #     args=training_args,
        #     model=train_model,
        #     train_dataset=data_train,
        #     eval_dataset=data_test,
        # )
        # trainer.set_patch_coordinator(mpc)
        # 0/0

        outputs = []

        print("evaluating test set with beam search")
        print("label smoothing off for eval")
        train_model.label_smoothing = 0
        

        ref_file = os.path.join(args.eval_output_dir, "refs.txt")
        out_file = os.path.join(args.eval_output_dir, "beam_out.txt")
        eval_write(data_test, train_model, ref_file, out_file, args.quiet)


    if args.eval_beam_speed:
        # python main.py --epochs=3 --batch_size=1 --train_samples=200 --valid_samples=5 --tokenizer_path=EleutherAI/gpt-neo-125M --model_path=EleutherAI/gpt-neo-125M --output_dir=./output --eval_beam_speed   --model_type=gpt-neo --prune_then_train --prune_leftover=1 --dense_pruning_method=topK --dense_pruning_submethod=1d_alt --mask_init=uniform --mask_init_scale=1
        import time
        train_model.eval()
        print("evaluating test set with beam search")
        print("label smoothing off for eval")
        train_model.label_smoothing = 0


        time_per_tok = 0
        for idx, sample in enumerate(data_test):
            input_ids = sample["input_ids"].to(device)

            
            # start = time.time()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            test_beam_outputs = train_model.generate(
                input_ids,
                max_length=511,  # up to the mask size
                num_beams=1,
                use_cache=True,
            )
            # out = train_model(input_ids)
            # end = time.time()
            end.record()
            torch.cuda.synchronize()

            forward_time = start.elapsed_time(end)

            print(test_beam_outputs.shape)
            print("idx", idx, "out shape", test_beam_outputs.shape[1])
            time_per_tok += (forward_time) 

        time_per_tok = time_per_tok / len(data_test)
        print("ave time per tok", time_per_tok)


    if args.save_outputs:
        print("saving training outputs")
        train_outputs = {}
        total = len(data_train)
        for idx, inputs in enumerate(data_train):
        # for idx, input_id in enumerate(samples):
            # inputs = wikisql_test[input_id]
            input_ids = inputs["input_ids"].cuda()
            new_inputs = {}
            with torch.no_grad():
                out = train_model(input_ids)["logits"]

                probs = torch.softmax(out, dim=-1).detach().cpu()

                topk_probs, topk_indices = torch.topk(out, k=20, dim=-1)

                new_inputs["distil_topk_probs"] = topk_probs
                new_inputs["distil_topk_indices"] = topk_indices

            context, completion = data_train.get_ctx_completion(idx)
            new_inputs["context"] = context
            new_inputs["completion"] = completion
            

            train_outputs[idx] = new_inputs

            if idx % 100 == 0: print(f"{idx} out of {total}")

        print("saving to file...")
        with open(os.path.join(args.output_dir, "train_teacher_outputs.pkl"), "wb") as handle:
            pickle.dump(train_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        # print("saving validation outputs")
        val_outputs = {}
        total = len(data_validation)
        for idx, inputs in enumerate(data_validation):
            input_ids = inputs["input_ids"].cuda()
            new_inputs = {}
            with torch.no_grad():
                out = train_model(input_ids)["logits"]

                probs = torch.softmax(out, dim=-1).detach().cpu()

                topk_probs, topk_indices = torch.topk(out, k=20, dim=-1)

                new_inputs["distil_topk_probs"] = topk_probs
                new_inputs["distil_topk_indices"] = topk_indices
            
            context, completion = data_validation.get_ctx_completion(idx)
            new_inputs["context"] = context
            new_inputs["completion"] = completion
            
            val_outputs[idx] = new_inputs

            if idx % 100 == 0: print(f"{idx} out of {total}")

        print("saving to file...")
        with open(os.path.join(args.output_dir, "val_teacher_outputs.pkl"), "wb") as handle:
            pickle.dump(val_outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)


    if args.analyze:
        print("doing model analysis")

        if args.output_dir == None:
            output_dir = "./outputs"
        os.makedirs(output_dir, exist_ok=True)
        with open(  os.path.join(output_dir, "test.pkl"), "wb" ) as file:
            pickle.dump({"test": "test"}, file)

        print("Test file")

        if not args.unmodify_gpt2:
            train_model = optimize_model(train_model, "dense", clone=False, keep_dim_mode="1d_alt")
        else:   
            print("not modifying model - unmodify_gpt2")

        if not isinstance(train_model, nn.DataParallel):
            train_model = nn.DataParallel(train_model)

        layer_sensitivity = {}
        cos_sims = {}
        for n, mod in train_model.named_modules():
            if not isinstance(mod, (GPTNeoMLP, GPT2MLP)): continue
            if isinstance(mod.c_fc, Conv1D):
                layer_sensitivity[n] = mod.c_fc.weight.new_zeros(( mod.c_fc.weight.shape[1], ))
            else:
                layer_sensitivity[n] = mod.c_fc.weight.new_zeros(( mod.c_fc.weight.shape[0], ))

        
        if args.a_validation:
            print("analyzing using validation set!!")
            data_train = data_validation


        trainer = PruningTrainer(
            sparse_args=sparse_args,
            args=training_args,
            model=train_model,
            train_dataset=data_train,
            eval_dataset=data_validation,
            callbacks=[LogDfCallback]
        )


        bsz = args.batch_size
        correct = 0
        ppl = 0
        print("collecting outputs")
        samples = 0
        dataloader = trainer.get_train_dataloader()
        total_losses = []
        for step, p_inputs in enumerate(dataloader):
        #     print("TEMP DELETE ME", step)
        #     continue
        # 0/0
        # if False:
            train_model.zero_grad()
            inputs = trainer._prepare_inputs(p_inputs)
            outputs = train_model(**inputs)
            # print("outputs shape", outputs["logits"].shape)

            samples += int(outputs["logits"].shape[0])
            # idx = step

            device = "cuda:0"
            
            if args.task == "wikisql":
                logits = outputs["logits"].to(device)
                logits = logits[..., :-1, :].contiguous().to(device)
                labels = inputs["labels"]
                labels = labels[..., 1:].contiguous().to(device)
                logits = torch.argmax(logits, axis=-1)
                mask = inputs["label_mask"].to(device)
                # acc = ((logits[:] == labels[:])*mask).sum() == mask.sum()
                correct_labels = ((logits[:] == labels[:])*mask).sum() / (mask.sum())  # full acc
                # print("correct labels", correct_labels)
                
                acc = ((logits[:] == labels[:])*mask).sum(axis=1, keepdims=True)
                acc = (acc == mask.sum(axis=1, keepdims=True)).sum()
                correct += acc
            else:
                logits = outputs["logits"]
                logits = logits[..., :-1, :].contiguous()
                labels = inputs["labels"]
                labels = labels[..., 1:].contiguous()
                logits = torch.argmax(logits, axis=-1)



            loss = outputs["raw_loss"] if isinstance(outputs, dict) else outputs[0]
            total_losses.append(loss.flatten())

            ppl =  torch.exp(torch.cat(total_losses).mean())
            if samples % 10 == 0:
                print(samples, "running acc", correct / (samples), "ppl", ppl   )

            all_neuron_outputs_fc = outputs["all_neuron_outputs_fc"]

            cnt = 0 
            for n, mod in train_model.named_modules():
                if not isinstance(mod, (GPTNeoMLP, GPT2MLP)): continue
                out = all_neuron_outputs_fc[cnt].detach().clone()
                out.view((-1, out.shape[-1]))

                out_corr = out.T @ out
                out_norm = out.norm(dim=0) **2

                if not ((n + "running_c_fc_sum") in cos_sims.keys()): 
                    cos_sims[n + "running_c_fc_sum"] =  out_corr
                    cos_sims[n + "running_c_fc_norm"] =  out_norm
                elif True:
                    cos_sims[n + "running_c_fc_sum"] = cos_sims[n + "running_c_fc_sum"] + out_corr
                    cos_sims[n + "running_c_fc_norm"] = cos_sims[n + "running_c_fc_norm"] +  out_norm
                cnt += 1

            loss = outputs["loss"].mean()
            loss.backward()

            with torch.no_grad():
                seen = []
                for n, mod in train_model.named_modules():
                    if not isinstance(mod, (GPTNeoMLP, GPT2MLP)): continue
                    
                    if isinstance(mod.c_fc, Conv1D):
                        sensitivity = (mod.c_fc.weight * mod.c_fc.weight.grad).sum(dim=0)
                        if mod.c_fc.bias is not None:
                            sensitivity += mod.c_fc.bias * mod.c_fc.bias.grad
                        sensitivity += (mod.c_proj.weight * mod.c_proj.weight.grad).sum(dim=1)
                    else:
                        sensitivity = (mod.c_fc.weight * mod.c_fc.weight.grad).sum(dim=1)
                        if mod.c_fc.bias is not None:
                            sensitivity += mod.c_fc.bias * mod.c_fc.bias.grad
                        sensitivity += (mod.c_proj.weight * mod.c_proj.weight.grad).sum(dim=0)
                    sensitivity = torch.abs(sensitivity)
                    sensitivity[torch.isnan(sensitivity)] = 0.0
                    sensitivity[torch.isinf(sensitivity)] = 0.0

                    layer_sensitivity[n] = layer_sensitivity[n] + sensitivity
         
        df = []

        all_sims = {}
        all_sens = {}
        total_neurons = 0
        total_sens = 0
        total_similar = 0
        for n, mod in train_model.named_modules():
            if not isinstance(mod, (GPTNeoMLP, GPT2MLP)): continue
            
            layer_sensitivity[n] = layer_sensitivity[n] / samples

            running_c_fc_norm = cos_sims[n + "running_c_fc_norm"] 
            running_c_fc_sum =  cos_sims[n + "running_c_fc_sum"] 

            norm = running_c_fc_norm ** .5
            norm = norm.view(-1, 1)
            sim = running_c_fc_sum /  (  norm * norm.T + 1e-6   )


            sim_upper = torch.triu(sim, diagonal=1)
            sim_once = sim_upper

            sim_cnt = ((sim_upper >= .8).sum(axis=1) > 0).sum()
            similar = sim_cnt / sim_upper.sum(axis=1).numel()

            print(n, "layer sens", layer_sensitivity[n].mean())
            print(n, similar.item(), "shape", sim.shape[0], "sim upper numel", sim_upper.sum(axis=1).numel() ) # make sure model is compiled with sim shape

            all_sims[n] = sim_once.clone().cpu()
            all_sens[n] = layer_sensitivity[n].clone().cpu() 

            total_similar += sim_cnt.item()
            total_neurons += sim_upper.sum(axis=1).numel() 
            total_sens += layer_sensitivity[n].sum().item()

        df.append({ 
                    "model_path": args.model_path, 
                    "total_neurons": total_neurons,
                    "sens_sum": total_sens, 
                    "similar": total_similar / total_neurons,
                    "unique": 1 - total_similar / total_neurons,
                    "a_leftover": args.a_leftover,
                    "a_distil": args.a_distil,
                    "a_method": args.a_method,
                    "ppl": torch.exp(torch.cat(total_losses).mean()),
                    "accuracy": correct / samples
        })

        print("final results")
        print(df[0])

        print("saving outputs...")

        total_sens_sim = {"model_path": args.model_path, "all_sens": all_sens, "all_sim": all_sims}
        with open(  os.path.join(output_dir, "all_sim_sens.pkl"), "wb" ) as file:
            pickle.dump(total_sens_sim, file)

        pd.DataFrame(df).to_pickle( os.path.join(output_dir, "results.df") )
        print("Files saved")

    print("done")