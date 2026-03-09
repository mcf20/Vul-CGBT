import logging
import sys

import datasets
import torch
import transformers
from datasets import load_dataset  
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed 
import torch
# local package
from arguments import build_args  # isort: skip

import os  
os.environ["WANDB_DISABLED"] = "true"  
logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def set_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = transformers.logging.INFO
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def main():
    model_args, data_args, training_args = build_args()

    set_seed(training_args.seed)
    set_logging()

    is_distributed_train = bool(training_args.local_rank != -1)
    logger.info(
        f"*** Process rank: {training_args.local_rank}, \
            device: {training_args.device}, \
            n_gpu: {training_args.n_gpu}, \
            distributed training: {is_distributed_train}, \
            fp16 training: {training_args.fp16}, \
            bf16 training: {training_args.fp16},"
    )
    logger.info(f"*** Model parameters {model_args}")
    logger.info(f"*** Data parameters {data_args}")
    logger.info(f"*** Training/evaluation parameters {training_args}")

    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )

    # Load datasets
    train_dataset = None
    eval_dataset = None

    with training_args.main_process_first(desc="build sft_dataset"):
        # 加载JSON数据集  
        dataset = load_dataset('json', data_files={'train': data_args.dataset_name_or_path}, split='train', cache_dir=data_args.cache_dir)
        # Tokenize数据集  
        def tokenize_function(examples):  
            return tokenizer(examples['code'], padding='max_length', truncation=True, max_length=data_args.block_size)  
        
        train_dataset = dataset.map(tokenize_function, num_proc=data_args.preprocessing_num_workers, batched=False, remove_columns=["code"], load_from_cache_file=False) 
        n_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        logger.info(f"*** builded train samples, total: {n_train_samples}")
        logger.info(f"*** train_dataset example: {train_dataset[1]}")

    # Load init model
    logger.info("*** Loading pretrained model for LLM")
    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    logger.info(f"*** torch_dtype {str(torch_dtype)}")
    model = RobertaForMaskedLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        trust_remote_code=True,
    )
    logger.info("*** Loaded pretrained model to cpu")

    # Initialize the Trainer
    # 数据收集器用于动态掩码  
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)  

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training loop
    if training_args.do_train:
        logger.info("*** Train")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = n_train_samples

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
