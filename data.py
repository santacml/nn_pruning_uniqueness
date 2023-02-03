from curses import raw
from typing import Dict
from transformers import AutoTokenizer
from torch.utils.data import Dataset
# from datasets import load_dataset, load_from_disk
import json
import torch
import pickle, os
from itertools import chain

import random


class ListDataset(Dataset):
    def __init__(self, dataset,
                       type_path: str,
                       input_length: int, 
                       output_length: int,
                       num_samples: int = None,
                       tokenizer = None, 
                       lower: bool = False) -> None:      

        self.dataset = list(dataset.values())
        self.type_path = type_path

        if num_samples:
            self.dataset = self.dataset[:num_samples]
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.lower = lower
        
        # self.tokenizer.pad_token = self.tokenizer.eos
        self.tokenizer.pad_token = "<|endoftext|>"
        # self.tokenizer.pad_token = 0

        self.preprocess()

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

        
    
    def clean_text(self, text: str) -> str:
        if self.lower:
            return text.lower()
        else:
            return text

    def preprocess(self):
        print("Preprocessing...")
        for n in range(len(self.dataset)):
            distil_topk_probs = self.dataset[n]["distil_topk_probs"]
            distil_topk_indices = self.dataset[n]["distil_topk_indices"]
            source, src_mask, targets, target_mask, context, completion = self.convert_to_features(self.dataset[n])
            self.dataset[n] = [source, src_mask, targets, target_mask, context, completion, distil_topk_probs, distil_topk_indices]
        print("Done.")
    

    
    def convert_to_features(self, example_batch):                
        # context, completion = example_batch
        context = example_batch["context"]
        completion = example_batch["completion"]
        context, completion = self.clean_text(context), self.clean_text(completion)

        inputs = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context  + completion], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        # query = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context], max_length=self.input_length, 
        #                                             padding='max_length', truncation=True, return_tensors="pt")
        query = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context], return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context + completion], max_length=self.output_length, 
        # targets = self.tokenizer.batch_encode_plus([input_ + target_], max_length=self.output_length, 
        # targets = self.tokenizer.batch_encode_plus(["S " + target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")

        input_ids = inputs["input_ids"].squeeze()
        query_ids = query["input_ids"]
        target_ids = targets["input_ids"].squeeze()

        
        context_encode = self.tokenizer.batch_encode_plus([context], return_tensors="pt")["input_ids"] # no bos as it gets shifted anyway
        context_encode_len = context_encode.shape[1]
        target_mask = torch.zeros(target_ids.shape[0] - 1)
        target_mask[context_encode_len:] = 1

        # same_toks = target_ids == query_ids
        # target_ids[same_toks] = target_ids[-1]  # keep only completion at the end
        # target_mask = target_ids - query_ids

        
        if self.type_path == "test":
            src_mask = query["attention_mask"].squeeze()
            return query_ids, src_mask, target_ids, target_mask, context, completion
        else:
            src_mask = inputs["attention_mask"].squeeze()
            return input_ids, src_mask, target_ids, target_mask, context, completion
        
        # encoded_input = self.tokenizer(input_, target_, max_length=self.output_length, padding="max_length", truncation=True, return_tensors="pt")
       
        # return encoded_input
  
    def __getitem__(self, index: int) -> dict:
        # source, targets = self.convert_to_features(self.dataset[index])
        
        # source_ids = source["input_ids"].squeeze()
        # target_ids = targets["input_ids"].squeeze()

        # src_mask    = source["attention_mask"].squeeze()
        # target_mask = targets["attention_mask"].squeeze()

        # return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids}


        source_ids, src_mask, target_ids, target_mask, context, completion, distil_topk_probs, distil_topk_indices = self.dataset[index]
        return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids, "label_mask": target_mask, "distil_topk_probs": distil_topk_probs, "distil_topk_indices": distil_topk_indices}


class e2e_nlg(Dataset): 
    def __init__(self, dataset_loader: str,
                       type_path: str,
                       input_length: int, 
                       output_length: int,
                       seed: int,
                       num_samples: int = None,
                       tokenizer = None, 
                       lower: bool = False) -> None:      

        self.stored_mr_ref = []
        self.dataset_loader = dataset_loader
        self.type_path = type_path

        
        # random.shuffle(dataset_loader)
        self.dataset_loader = dataset_loader.shuffle(seed=seed)

        if num_samples:
            self.dataset_loader = self.dataset_loader.select(range(num_samples))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        
        # self.tokenizer.pad_token = self.tokenizer.eos
        self.tokenizer.pad_token = "<|endoftext|>"
        # self.tokenizer.pad_token = 0

        self.preprocess()
  
    def __len__(self) -> int:
        # return self.dataset.shape[0]
        return len(self.dataset_loader)
        # return len(self.dataset["train"])
    
    def clean_text(self, text: str) -> str:
        return text

    def preprocess(self):
        print("Preprocessing...")
        def convert(example):
            # return self.convert_to_features([example["meaning_representation"], example["human_reference "]])
            # 0/0
            # source_ids, src_mask, target_ids, target_mask = self.convert_to_features([example["meaning_representation"], example["human_reference"]])
            
            context, completion = [example["meaning_representation"], example["human_reference"]]
            context, completion = self.clean_text(context), self.clean_text(completion)

            # print(context + " " + completion)
            # 0/0

            inputs = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context + " " + completion], max_length=self.input_length, 
                                                        padding='max_length', truncation=True, return_tensors="pt")

            query = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context + " " ], return_tensors="pt")
            
            targets = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context + " " + completion], max_length=self.output_length, 
                                                        padding='max_length', truncation=True, return_tensors="pt")

            input_ids = inputs["input_ids"].squeeze()
            query_ids = query["input_ids"]
            target_ids = targets["input_ids"].squeeze()

            
            context_encode = self.tokenizer.batch_encode_plus([context], return_tensors="pt")["input_ids"] # no bos as it gets shifted anyway
            context_encode_len = context_encode.shape[1]
            target_mask = torch.zeros(target_ids.shape[0] - 1)
            target_mask[context_encode_len:] = 1
            
            if self.type_path == "test":
                src_mask = query["attention_mask"].squeeze()
                source_ids, src_mask, target_ids, target_mask =  query_ids, src_mask, target_ids, target_mask
            else:
                src_mask = inputs["attention_mask"].squeeze()
                source_ids, src_mask, target_ids, target_mask =  input_ids, src_mask, target_ids, target_mask

            example["input_ids"] = source_ids
            example["attention_mask"] = src_mask
            example["labels"] = target_ids
            example["label_mask"] = target_mask
            example["context"] = example["meaning_representation"]
            example["completion"] = example["human_reference"]
            return example

        self.dataset_loader = self.dataset_loader.map(convert)

        print("Done.")
    
    def convert_to_features(self, example_batch):
        context, completion = example_batch
        context, completion = self.clean_text(context), self.clean_text(completion)

        inputs = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context + " " + completion], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")

        query = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context + " " ], return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context + " " + completion], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")

        input_ids = inputs["input_ids"].squeeze()
        query_ids = query["input_ids"]
        target_ids = targets["input_ids"].squeeze()

        
        context_encode = self.tokenizer.batch_encode_plus([context], return_tensors="pt")["input_ids"] # no bos as it gets shifted anyway
        context_encode_len = context_encode.shape[1]
        target_mask = torch.zeros(target_ids.shape[0] - 1)
        target_mask[context_encode_len:] = 1

        # same_toks = target_ids == query_ids
        # target_ids[same_toks] = target_ids[-1]  # keep only completion at the end
        # target_mask = target_ids - query_ids


        if self.type_path == "test":
            src_mask = query["attention_mask"].squeeze()
            return query_ids, src_mask, target_ids, target_mask
        else:
            src_mask = inputs["attention_mask"].squeeze()
            return input_ids, src_mask, target_ids, target_mask
        
        # encoded_input = self.tokenizer(input_, target_, max_length=self.output_length, padding="max_length", truncation=True, return_tensors="pt")
       
        # return encoded_input
  
    def __getitem__(self, index: int) -> dict:
        # print(self.dataset_loader[index])
        # 0/0
        # ex = self.dataset_loader[index]
        return  self.dataset_loader[index]
        
    # def get_ctx_completion(self, index: int) -> dict:
    #     example = self.orig_dataset_loader[index]
        # return {"context": example["meaning_representation"], "completion": example["human_reference"]}


class wikitext(Dataset): 
    def __init__(self, dataset_loader: str,
                       dataset_path: str,
                       type_path: str,
                       input_length: int, 
                       output_length: int,
                       num_samples: int = None,
                       tokenizer = None, 
                       lower: bool = False) -> None:      

        if dataset_path is None and dataset_loader is None:
            raise ValueError("dataset_path or dataset_loader must be specified")


        self.stored_mr_ref = []
        self.dataset_loader = dataset_loader
        self.type_path = type_path

        if num_samples:
            self.dataset_loader = self.dataset_loader.select(range(num_samples))
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        
        # self.tokenizer.pad_token = self.tokenizer.eos
        self.tokenizer.pad_token = "<|endoftext|>"
        # self.tokenizer.pad_token = 0

        self.preprocess()
  
    def __len__(self) -> int:
        # return self.dataset.shape[0]
        return len(self.dataset_loader)
        # return len(self.dataset["train"])
    
    def clean_text(self, text: str) -> str:
        return text

    def preprocess(self):
        print("Preprocessing...")

        def convert(example):
            # return self.convert_to_features([example["meaning_representation"], example["human_reference "]])
            # 0/0
            # source_ids, src_mask, target_ids, target_mask = self.convert_to_features([example["meaning_representation"], example["human_reference"]])
            
            context = example["text"]
            context = self.clean_text(context)


            # print(context + " " + completion)
            # 0/0

            # inputs = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context ], max_length=None, truncation=False, return_tensors="pt")
            inputs = self.tokenizer.batch_encode_plus([context ], max_length=self.input_length, padding='max_length', truncation=True, return_tensors="pt")

            input_ids = inputs["input_ids"].squeeze()
            target_ids = inputs["input_ids"].squeeze()

            
            context_encode = self.tokenizer.batch_encode_plus([context], return_tensors="pt")["input_ids"] # no bos as it gets shifted anyway
            context_encode_len = context_encode.shape[1] if context_encode.shape[1] < self.input_length else self.input_length
            target_mask = torch.zeros(target_ids.shape[0] - 1)
            target_mask[context_encode_len:] = 1
            
            src_mask = inputs["attention_mask"].squeeze()
            source_ids, src_mask, target_ids, target_mask =  input_ids, src_mask, target_ids, target_mask

            example["input_ids"] = source_ids
            example["attention_mask"] = src_mask
            example["labels"] = target_ids
            example["label_mask"] = target_mask

            return example

        self.dataset_loader = self.dataset_loader.map(convert)

        print("Done.")

    
    def __getitem__(self, index: int) -> dict:
        # print(self.dataset_loader[index])
        # 0/0
        # ex = self.dataset_loader[index]
        return  self.dataset_loader[index]
        

class wikisql_lora(Dataset): 
    ''' stolen from LoRA repo, with modifications'''
    def __init__(self, dataset_path: str,
                       type_path: str,
                       input_length: int, 
                       output_length: int,
                       num_samples: int = None,
                       tokenizer = None, 
                       lower: bool = False) -> None:      

        self.ft_file = dataset_path
        self.dataset = self.read_ft_file(dataset_path)
        self.type_path = type_path

        if num_samples:
            self.dataset = self.dataset[:num_samples]
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.lower = lower
        
        # self.tokenizer.pad_token = self.tokenizer.eos
        self.tokenizer.pad_token = "<|endoftext|>"
        # self.tokenizer.pad_token = 0

        self.preprocess()
  
    def __len__(self) -> int:
        # return self.dataset.shape[0]
        return len(self.dataset)
        # return len(self.dataset["train"])
    
    def clean_text(self, text: str) -> str:
        if self.lower:
            return text.lower()
        else:
            return text

    def preprocess(self):
        print("Preprocessing...")
        for n in range(len(self.dataset)):
            source, src_mask, targets, target_mask, context, completion = self.convert_to_features(self.dataset[n])
            self.dataset[n] = [source, src_mask, targets, target_mask, context, completion]
        print("Done.")
    
    def convert_to_features(self, example_batch):                
        context, completion = example_batch
        context, completion = self.clean_text(context), self.clean_text(completion)

        inputs = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context  + completion], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        # query = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context], max_length=self.input_length, 
        #                                             padding='max_length', truncation=True, return_tensors="pt")
        query = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context], return_tensors="pt")
        
        targets = self.tokenizer.batch_encode_plus(["<|endoftext|> " + context + completion], max_length=self.output_length, 
        # targets = self.tokenizer.batch_encode_plus([input_ + target_], max_length=self.output_length, 
        # targets = self.tokenizer.batch_encode_plus(["S " + target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")

        input_ids = inputs["input_ids"].squeeze()
        query_ids = query["input_ids"]
        target_ids = targets["input_ids"].squeeze()

        
        context_encode = self.tokenizer.batch_encode_plus([context], return_tensors="pt")["input_ids"] # no bos as it gets shifted anyway
        context_encode_len = context_encode.shape[1]
        target_mask = torch.zeros(target_ids.shape[0] - 1)
        target_mask[context_encode_len:] = 1

        # same_toks = target_ids == query_ids
        # target_ids[same_toks] = target_ids[-1]  # keep only completion at the end
        # target_mask = target_ids - query_ids

        
        if self.type_path == "test":
            src_mask = query["attention_mask"].squeeze()
            return query_ids, src_mask, target_ids, target_mask, context, completion
        else:
            src_mask = inputs["attention_mask"].squeeze()
            return input_ids, src_mask, target_ids, target_mask, context, completion
        
        # encoded_input = self.tokenizer(input_, target_, max_length=self.output_length, padding="max_length", truncation=True, return_tensors="pt")
       
        # return encoded_input
  
    def __getitem__(self, index: int) -> dict:
        # source, targets = self.convert_to_features(self.dataset[index])
        
        # source_ids = source["input_ids"].squeeze()
        # target_ids = targets["input_ids"].squeeze()

        # src_mask    = source["attention_mask"].squeeze()
        # target_mask = targets["attention_mask"].squeeze()

        # return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids}


        source_ids, src_mask, target_ids, target_mask, context, completion = self.dataset[index]
        return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids, "label_mask": target_mask}
        
    def get_ctx_completion(self, index: int) -> dict:
        source_ids, src_mask, target_ids, target_mask, context, completion = self.dataset[index]
        return {"context": context, "completion": completion}


    def read_ft_file(self, ft_file):
        ft_samples = []
        with open(ft_file, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                context = items['context']
                completion = items['completion']
                ft_samples.append([context, completion])
        random.shuffle(ft_samples)
        return ft_samples

class wikisql_lora_orig(Dataset):
    def __init__(self, dataset_path: str,
                       type_path: str,
                       input_length: int, 
                       output_length: int,
                       num_samples: int = None,
                       tokenizer = None, 
                       sql2txt: bool = True) -> None:      

        self.ft_file = dataset_path
        self.dataset = self.read_ft_file(dataset_path)



        if num_samples:
            self.dataset = self.dataset[:num_samples]
        self.input_length = input_length
        self.tokenizer = tokenizer
        self.output_length = output_length
        self.sql2txt = sql2txt
        
        # self.tokenizer.pad_token = self.tokenizer.eos
        self.tokenizer.pad_token = "<|endoftext|>"
  
    def __len__(self) -> int:
        # return self.dataset.shape[0]
        return len(self.dataset)
        # return len(self.dataset["train"])
    
    def clean_text(self, text: str) -> str:
        # return text.replace('\n','').replace('``', '').replace('"', '').lower()
        # return text.replace('\n','').replace('``', '').lower()
        return text.lower()

    
    def convert_to_features(self, example_batch):                
        if self.sql2txt:
            # sql to text
            0/0
            input_ = "translate SQL to English: " + self.clean_text(example_batch['sql']['human_readable'])
            target_ = self.clean_text(example_batch['question'])
        else: 
            context, completion = example_batch

            input_ = self.clean_text(context)
            target_ = self.clean_text(completion)
        
        # print(input_)
        # 0/0
        source = self.tokenizer.batch_encode_plus([input_], max_length=self.input_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")
        
        # targets = self.tokenizer.batch_encode_plus([target_], max_length=self.output_length, 
        targets = self.tokenizer.batch_encode_plus([input_ + target_], max_length=self.output_length, 
        # targets = self.tokenizer.batch_encode_plus(["S " + target_], max_length=self.output_length, 
                                                     padding='max_length', truncation=True, return_tensors="pt")

        return source, targets
        
        # encoded_input = self.tokenizer(input_, target_, max_length=self.output_length, padding="max_length", truncation=True, return_tensors="pt")
       
        # return encoded_input
  
    def __getitem__(self, index: int) -> dict:
        source, targets = self.convert_to_features(self.dataset[index])
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        # return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        # return {"input_ids": source_ids, "attention_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        return {"input_ids": source_ids, "attention_mask": src_mask, "labels": target_ids}

        # encoded_input =  self.convert_to_features(self.dataset[index])
        # output = {}
        # output["input_ids"] = encoded_input["input_ids"]
        # output["labels"] = encoded_input["input_ids"]
        # output["attention_mask"] = encoded_input["attention_mask"]
        # return output

    def read_ft_file(self, ft_file):
        ft_samples = []
        with open(ft_file, 'r') as reader:
            for line in reader:
                items = json.loads(line.strip())
                context = items['context']
                completion = items['completion']
                ft_samples.append([context, completion])
        return ft_samples



def get_dataset_wikisql_distil(tokenizer_path: str, dataset, type_path: str, num_samples: int, max_input_length, max_output_length, sql2txt, lower=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=max_output_length, add_special_tokens=True)

    return ListDataset( dataset=dataset,
                    type_path=type_path,
                    num_samples=num_samples,  
                    input_length=max_input_length, 
                    output_length=max_output_length,
                    tokenizer=tokenizer,
                    lower=lower)


def get_dataset(tokenizer_path: str, dataset_path: str, type_path: str, num_samples: int, max_input_length, max_output_length, sql2txt, lower=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=max_output_length, add_special_tokens=True)

    return wikisql_lora( dataset_path=dataset_path,
                    type_path=type_path,
                    num_samples=num_samples,  
                    input_length=max_input_length, 
                    output_length=max_output_length,
                    tokenizer=tokenizer,
                    lower=lower)


                    
def get_dataset_e2e_nlg(tokenizer_path: str, dataset_loader, type_path: str, num_samples: int, max_input_length, max_output_length, seed, sql2txt, lower=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=max_output_length, add_special_tokens=True)

    return e2e_nlg( dataset_loader=dataset_loader,
                    type_path=type_path,
                    num_samples=num_samples,  
                    input_length=max_input_length, 
                    output_length=max_output_length,
                    seed=seed,
                    tokenizer=tokenizer,
                    lower=lower)


def get_datasets_samsum(tokenizer_path, raw_dataset, max_output_length, num_samples, is_test=False):
    if num_samples is not None:
        raw_dataset = raw_dataset.select(range(num_samples))

    column_names = raw_dataset.column_names

    max_input_length = max_output_length


    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=max_output_length, add_special_tokens=True)
 
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.sep_token = "<sep>"




    def tokenize_function(examples):
        
        if is_test:
            output = tokenizer(examples["dialogue"] + tokenizer.sep_token,  truncation=True, return_tensors="pt")
            context_encode = tokenizer(examples["dialogue"] + tokenizer.sep_token) # no bos as it gets shifted anyway
        else:
            output = tokenizer(examples["dialogue"] + tokenizer.sep_token + examples["summary"], padding='max_length', truncation=True, return_tensors="pt")
            context_encode = tokenizer(examples["dialogue"] + tokenizer.sep_token + examples["summary"]) # no bos as it gets shifted anyway
        context_encode_len = len(context_encode["input_ids"])
        # print(context_encode_len)
        target_mask = torch.zeros(output["input_ids"].shape[1] - 1)
        target_mask[context_encode_len:] = 1

        # print(target_mask)

        output["target_mask"] = target_mask
        output["labels"] = output["input_ids"].clone()
        # output["context"] = examples["dialogue"]
        output["completion"] = examples["summary"]

        return output

    
    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=False,
        num_proc=3,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    return tokenized_datasets


 


def get_datasets_wikitext(tokenizer_path: str, raw_datasets, max_output_length, num_samples, block_size, seed=1234, is_analyze=False):

    # adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py

    if num_samples is not None:
        raw_datasets = raw_datasets.select(range(num_samples))

    if not is_analyze:  # want to keep analysis the same, as chunking could go differently if shuffled...
        raw_datasets = raw_datasets.shuffle(seed=seed)
    else:
        print("NOT SHUFFLING WIKITEXT DATA - ANALYZE MODE")

    # raw_datasets = load_dataset(dataset_path, name='wikitext-103-v1', split=['train', 'validation', 'test'])

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=max_output_length, add_special_tokens=True)


    column_names = raw_datasets.column_names
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # if training_args.do_train:
    #     column_names = raw_datasets["train"].column_names
    # else:
    #     column_names = raw_datasets["validation"].column_names

    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    # tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        # with CaptureLogger(tok_logger) as cl:
        output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        # if "Token indices sequence length is longer than the" in cl.out:
        #     tok_logger.warning(
        #         "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
        #         " before being passed to the model."
        #     )
        return output

    # with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=3,
        remove_columns=column_names,
        # load_from_cache_file=not data_args.overwrite_cache,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    if block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            # logger.warning(
            #     f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
            #     "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            # )
            block_size = 1024
    else:
        # if data_args.block_size > tokenizer.model_max_length:
            # logger.warning(
            #     f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
            #     f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            # )
        block_size = min(block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=3,
        # load_from_cache_file=not data_args.overwrite_cache,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets