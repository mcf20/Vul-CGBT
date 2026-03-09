from typing import Dict

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, Trainer, set_seed
from datasets import Dataset, load_dataset, DatasetDict   
from torch.utils.data import DataLoader, random_split 
from tqdm import tqdm 
import random  


def convert_example_to_features_for_cl(example, tokenizer, max_length: int):
    IGNORE_INDEX = -100

    if "code" in example and "positive" in example and "negatives" in example:
        positive = example["positive"]
        code = example["code"]
        negative = random.choice(example["negatives"])  

        sentence = tokenizer([code], max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True, add_special_tokens=False)
        sentence['input_ids'] = [[tokenizer.bos_token_id] + input_ids for input_ids in sentence['input_ids']]
        sentence['input_ids'] = torch.tensor(sentence['input_ids'], dtype=torch.int)
        positive = tokenizer([positive], max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True, add_special_tokens=False)
        positive['input_ids'] = [[tokenizer.bos_token_id] + input_ids for input_ids in positive['input_ids']]
        positive['input_ids'] = torch.tensor(positive['input_ids'], dtype=torch.int)
        negative = tokenizer([negative], max_length=max_length - 1, return_attention_mask=False, padding=False, truncation=True, add_special_tokens=False)
        negative['input_ids'] = [[tokenizer.bos_token_id] + input_ids for input_ids in negative['input_ids']]
        negative['input_ids'] = torch.tensor(negative['input_ids'], dtype=torch.int)

    else:
        raise ValueError("Invalid example")

    features = {
        "sentence": sentence["input_ids"],
        "positive": positive["input_ids"],
        "negative": negative["input_ids"],
    }

    return features


def build_dataset(
    dataset_name_or_path: str,
    split_name: str,
    tokenizer,
    num_proc: int = 1,
    block_size: int = 1024,
    cache_dir: str = None,
    load_from_cache_file=False,
) -> Dataset:

    raw_dataset: Dataset = load_dataset("json", data_files=[dataset_name_or_path], split=split_name, cache_dir=cache_dir)
    #column_names = list(raw_dataset.features)  

    sft_dataset = raw_dataset.map(
        convert_example_to_features_for_cl,
        fn_kwargs={"tokenizer": tokenizer, "max_length": block_size},
        batched=False,
        num_proc=num_proc,
        remove_columns=None,
        load_from_cache_file=load_from_cache_file,
    )
    return sft_dataset
