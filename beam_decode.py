# from tkinter import N
# from numpy.core.fromnumeric import size
import torch
import datasets
import transformers
from transformers import TrainerCallback
import os
import argparse
from nn_pruning.sparse_trainer import SparseTrainer, TimingModule
from datasets import load_dataset
from datasets import load_metric
from data import get_dataset, get_dataset_e2e_nlg
from transformers import TrainingArguments
import torch 
from nn_pruning.patch_coordinator import ModelPatchingCoordinator
from nn_pruning.inference_model_patcher import optimize_model
from model import GPTNeoForCausalLM, GPT2LMHeadModel, GPTNeoMLP
import numpy as np
from torch import nn
import pandas as pd
from utils import PruningTrainer, args_to_hparams
import os
import evaluate

def setup():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"
    DISABLE_MLFLOW_INTEGRATION = True

    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='PyTorch GPT-Neo ft script')


    parser.add_argument('--eval_output_dir', default="./", help='location of saved beam output')

    parser.add_argument('--seed', default=0, type=int, help='seed')

    parser.add_argument('--task', default="wikisql", help='which task to run', choices=('wikisql', 'e2e_nlg'))
    parser.add_argument('--dataset_path', default="/home/azureuser/FT_wikisql_v8/", help='location of data corpus')
    parser.add_argument('--tokenizer_path', required=True,  help='location of tokenizer')

    parser.add_argument('--model_path', required=True, help='location of model')
    parser.add_argument('--token_max_len', default=512, type=int, help='token max len')

    parser.add_argument('--valid_samples', default=None, type=int, help='number of validation samples to use')


    parser.add_argument('--quiet', action='store_true', help='no prints')

    return parser


def eval_rouge(data_test, tokenizer, train_model, quiet=False):
    rouge = evaluate.load('rouge')
    rouge_predictions = {}
    rouge_references = []
    
    refs = {}
    outputs = []

    print("evaluating test set with beam search")
    
    accs = []
    num_samples = len(data_test)
    # for idx, inputs in enumerate(data_test):
    for idx in range(num_samples):
        if idx % 50 == 0:
            print("BEAM idx", idx, "out of", num_samples)
        inputs = data_test[idx]
        inputs["input_ids"] = torch.tensor(inputs["input_ids"])
        inputs["target_mask"] = torch.tensor(inputs["target_mask"])



        rouge_references
        # orig_ref = data_test.get_ctx_completion(idx)
        # ctx = inputs["context"]
        completion = inputs["completion"]
        rouge_references.append(completion)
        if not (completion in rouge_predictions):
            rouge_predictions[completion] = []

        


        # print(inputs)
        # 0/0
        # print(inputs["input_ids"])
        # input_ids = torch.tensor(inputs["input_ids"]).cuda()
        input_ids = inputs["input_ids"].cuda()
        # print(input_ids)
        inputs_decode = tokenizer.decode(input_ids[0])

        if not quiet:
            # print("context")
            # print(ctx)
            # print("inputs_decode")
            # print(inputs_decode)
            print("Example completion")
            print(completion)

        test_beam_outputs = train_model.generate(
            input_ids,
            max_length=1023,  # up to the mask size
            num_beams=10,
            length_penalty=.9,
            no_repeat_ngram_size=4
        )

        logits = test_beam_outputs[-1, :].squeeze()
        
        output = tokenizer.decode(logits)
        output = output[output.find("<sep>"): ].replace("<sep>", "").replace("<|endoftext|>", "")

        rouge_predictions[completion].append(output)

        if not quiet:
            # print()
            print("out decode")
            print(output)
            # print()
            # print()
    
    ordered_references = []
    ordered_predictions = []
    for ref in rouge_references:
        for pred in rouge_predictions[ref]:
            ordered_references.append(ref)
            ordered_predictions.append(pred)


    results = rouge.compute(predictions=ordered_predictions,
                            references=rouge_references)
    print(results)




def eval_write(data_test, train_model, ref_file, out_file, quiet=False):
    refs = {}
    outputs = []

    print("evaluating test set with beam search")
    
    accs = []
    num_samples = len(data_test)
    # for idx, inputs in enumerate(data_test):
    for idx in range(num_samples):
        if idx % 50 == 0:
            print("BEAM idx", idx, "out of", num_samples)
        inputs = data_test[idx]
        # print(inputs)
        # 0/0
        inputs["input_ids"] = torch.tensor(inputs["input_ids"])
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"])
        inputs["labels"] = torch.tensor(inputs["labels"])
        label_mask = False
        if "label_mask" in inputs.keys():
            label_mask = True
            inputs["label_mask"] = torch.tensor(inputs["label_mask"])

        # orig_ref = data_test.get_ctx_completion(idx)
        ctx = inputs["context"]
        completion = inputs["completion"].replace("\n", "").replace("<|endoftext|>", "")
        if ctx in refs:
            refs[ctx]["references"].append(completion)
            continue
        else:
            refs[ctx] = {"references": [completion]}

        # print(inputs)
        # 0/0
        # print(inputs["input_ids"])
        # input_ids = torch.tensor(inputs["input_ids"]).cuda()
        input_ids = inputs["input_ids"].cuda()
        # print(input_ids)
        inputs_decode = data_test.tokenizer.decode(input_ids[0])

        if not quiet:
            print("context")
            print(ctx)
            print("inputs_decode")
            print(inputs_decode)

        test_beam_outputs = train_model.generate(
            input_ids,
            max_length=511,  # up to the mask size
            num_beams=10,
            length_penalty=.9,
            no_repeat_ngram_size=4
        )

        logits = test_beam_outputs[-1, :].squeeze()
        labels = inputs["labels"][:logits.shape[0]].cuda().squeeze()
        if label_mask:
            mask = inputs["label_mask"][:logits.shape[0]].cuda().squeeze()

            acc = ((logits[:] == labels[:])*mask).sum() == mask.sum()

            accs.append(acc)

            output = logits[mask == 1]

        
        output = data_test.tokenizer.decode(output)
        output = output.replace("\n", "").replace("<|endoftext|>", "")[3:]
        outputs.append(output)

        refs[ctx]["output"] = outputs[-1]

        if not quiet:
            print()
            print("out decode")
            print(outputs[-1])
            print()
            print()

    if label_mask:
        acc = float(sum(accs) / len(accs))
        print("Testset BEAM accuracy", acc)
        
    with open(ref_file, 'w', encoding='utf8') as ref_writer, \
             open(out_file, 'w', encoding='utf8') as out_writer:

        for key, ref in refs.items():
            ref_writer.writelines([sample_ref + "\n" for sample_ref in ref["references"]])
            ref_writer.write("\n")

            out_writer.write(ref["output"])
            out_writer.write("\n")





if __name__ == "__main__": 
    parser = setup()

    args = parser.parse_args()
    parser_args = parser.parse_args()
    print("arguments")
    print(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    datasets.logging.set_verbosity_error()
    transformers.logging.set_verbosity_error()
    print(f"Using transformers v{transformers.__version__} and datasets v{datasets.__version__} and torch v{torch.__version__}")

    model_name = args.model_path

    if args.task == "wikisql":
        # data_train = get_dataset(args.tokenizer_path, os.path.join(args.dataset_path, "train.jsonl"), "train", args.train_samples, args.token_max_len, args.token_max_len, False, lower=False)
        # data_validation = get_dataset(args.tokenizer_path,os.path.join(args.dataset_path, "validation.jsonl"), "validation", args.valid_samples, args.token_max_len, args.token_max_len, False, lower=False)
        data_test = get_dataset(args.tokenizer_path, os.path.join(args.dataset_path, "test.jsonl"), "test", args.valid_samples, args.token_max_len, args.token_max_len, False, lower=False)
    elif args.task == "e2e_nlg":
        data_train, data_validation, data_test = load_dataset(args.dataset_path, split=['train', 'validation', 'test'])

        # data_train = get_dataset_e2e_nlg(args.tokenizer_path, data_train, "train", args.train_samples, args.token_max_len, args.token_max_len, False, lower=False)
        # data_validation = get_dataset_e2e_nlg(args.tokenizer_path,data_validation, "validation", args.valid_samples, args.token_max_len, args.token_max_len, False, lower=False)
        data_test = get_dataset_e2e_nlg(args.tokenizer_path, data_test, "test", args.valid_samples, args.token_max_len, args.token_max_len, False, lower=False)
    else:
        print("not implemented")
        0/0


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)

    ref_file = os.path.join(args.eval_output_dir, "refs.txt")
    out_file = os.path.join(args.eval_output_dir, "beam_out.txt")
    eval_write(data_test, train_model, ref_file, out_file, args.quiet)

        
    print("done")