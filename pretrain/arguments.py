import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import transformers
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("The model checkpoint for weights initialization. Don't set if you want to train a model from scratch.")},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name or local path of the dataset to use"},
    )
    eval_dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The name or local path of the dataset to use"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": ("For debugging purposes or quicker training.")},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": ("For debugging purposes or quicker training.")},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    cache_dir: str = field(  
        default=None,  
        metadata={"help": "Path to a directory where the model will be cached."},  
    ) 
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    Arguments related to the training process itself.
    For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """

    logging_first_step: Optional[bool] = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    do_eval: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to do eval.")},
    )
    do_rdrop: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether to do rdrop.")},
    )
    evaluation_strategy: Optional[str] = field(
        default="no",
        metadata={"help": ("The evaluation strategy to adopt during training.")},
    )
    eval_steps: Optional[int] = field(
        default=0,
        metadata={"help": ("Run evaluation every X steps.")},
    )
    load_best_model_at_end: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether or not to load the best model found during training at the end of training.")},
    )
    metric_for_best_model: Optional[str] = field(
        default="eval_loss",
        metadata={"help": ("The metric to use to compare two different models.")},
    )
    greater_is_better: Optional[bool] = field(
        default=False,
        metadata={"help": ("Whether the `metric_for_best_model` should be maximized or not.")},
    )
    optim: Optional[str] = field(default="adamw_torch")


class ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats, and bools (default to strings)
                    if base_type in [int, float, bool]:
                        inputs[arg] = base_type(val)

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs


def build_parser() -> ArgumentParser:
    parser = ArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    return parser


def build_args():
    parser = build_parser()
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # parse command line yaml only
        model_args, data_args, training_args = parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
        # parse command line args and yaml file
        model_args, data_args, training_args = parser.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
    else:
        # parse command line args only
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args
