import logging
import sys

import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import RobertaTokenizer, RobertaModel, Trainer, set_seed
from transformers import TrainerCallback  
from itertools import takewhile
import os

# 设置自定义缓存目录（在导入其他模块之前设置）
custom_cache_dir = "/data/huggingface_cache"  # 请修改为您的大容量磁盘路径
os.makedirs(custom_cache_dir, exist_ok=True)
os.environ['HF_DATASETS_CACHE'] = os.path.join(custom_cache_dir, 'datasets')
os.environ['TRANSFORMERS_CACHE'] = os.path.join(custom_cache_dir, 'transformers')

from dataset_cl_with_neg import build_dataset
from infonce_loss import InfoNCE
import copy

import os  
os.environ["WANDB_DISABLED"] = "true"  
logger = logging.getLogger(__name__)
  
def set_logging():  
    logging.basicConfig(  
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  
        datefmt="%Y-%m-%d %H:%M:%S",  
        handlers=[logging.StreamHandler(sys.stdout)],  
    )  
    log_level = logging.INFO  
    logger.setLevel(log_level)  
    datasets.utils.logging.set_verbosity(log_level)  
    transformers.utils.logging.set_verbosity(log_level)  
    transformers.utils.logging.enable_default_handler()  
    transformers.utils.logging.enable_explicit_format()  
  
class MomentumEncoderCallback(TrainerCallback):  
    def __init__(self, momentum_encoder, model, momentum=0.995):  
        super().__init__()  
        self.momentum_encoder = momentum_encoder  
        self.model = model  
        self.momentum = momentum  
  
    def on_step_end(self, args, state, control, **kwargs):  
        with torch.no_grad():  
            for param_q, param_k in zip(self.model.parameters(), self.momentum_encoder.parameters()):  
                if param_k.data.shape == param_q.data.shape:  
                    param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)  
  
class MyTrainer(Trainer):          
    def __init__(self, *args, model_args=None, **kwargs):          
        super(MyTrainer, self).__init__(*args, **kwargs)      
        self.momentum_encoder = copy.deepcopy(self.model)   
        self.momentum_encoder.to(self.model.device)    
  
        # 冻结 momentum_encoder 的参数      
        for param in self.momentum_encoder.parameters():      
            param.requires_grad = False    
  
        # 初始化队列和其他参数      
        self.momentum = 0.995      
        self.queue_size = 256     
        self.temperature = 0.05    
  
        self.momentum_encoder.register_buffer(      
            'queue',      
            torch.randn(self.queue_size, self.model.config.hidden_size, device=self.model.device)      
        )      
        self.momentum_encoder.queue = F.normalize(self.momentum_encoder.queue, dim=1)    
        self.momentum_encoder.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long, device=self.model.device))    
  
        # 创建对缓冲区的引用      
        self.queue = self.momentum_encoder.queue      
        self.queue_ptr = self.momentum_encoder.queue_ptr        
  
    def first_token_pool(self, last_hidden_states, attention_mask):    
        # 检查是否是左侧填充    
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])    
        if left_padding:    
            # 左侧填充，取第一个 token    
            return last_hidden_states[:, 0]    
        else:    
            # 右侧填充，计算实际序列长度并获取最后一个非填充 token    
            sequence_lengths = attention_mask.sum(dim=1) - 1    
            batch_size = last_hidden_states.shape[0]    
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]    
  
    def compute_loss(self, model, inputs, return_outputs=False):      
        """      
        Compute contrastive loss including positive samples and negatives from the queue.  
        """  
        # Extract inputs for anchor and positive samples  
        sentence_inputs = {k.replace("sentence_", ""): v.to(self.model.device) for k, v in inputs.items() if "sentence_" in k}  
        positive_inputs = {k.replace("positive_", ""): v.to(self.model.device) for k, v in inputs.items() if "positive_" in k}
        negative_inputs = {k.replace("negative_", ""): v.to(self.model.device) for k, v in inputs.items() if "negative_" in k}
  
        sentence_outputs = model(**sentence_inputs, output_hidden_states=True)    
        positive_outputs = model(**positive_inputs, output_hidden_states=True)
        negative_outputs = model(**negative_inputs, output_hidden_states=True) 
        sentence_embs = self.first_token_pool(sentence_outputs.last_hidden_state, sentence_inputs["attention_mask"])
        positive_embs = self.first_token_pool(positive_outputs.last_hidden_state, positive_inputs["attention_mask"])   
        negative_embs = self.first_token_pool(negative_outputs.last_hidden_state, negative_inputs["attention_mask"])  
  
        # Normalize embeddings  
        sentence_embs = F.normalize(sentence_embs, p=2, dim=1)  
        positive_embs = F.normalize(positive_embs, p=2, dim=1)
        negative_embs = F.normalize(negative_embs, p=2, dim=1)
    
        # Use momentum encoder to compute embeddings for queue update  
        with torch.no_grad():  
            # Compute embeddings of current batch using momentum encoder  
            momentum_outputs = self.momentum_encoder(**sentence_inputs, output_hidden_states=True)  
            momentum_embs = self.first_token_pool(momentum_outputs.last_hidden_state, sentence_inputs["attention_mask"])  
            momentum_embs = F.normalize(momentum_embs, p=2, dim=1)  
  
        # Enqueue the embeddings, dequeue if necessary  
        self._dequeue_and_enqueue(momentum_embs)  
  
        # Compute similarities  
        pos_sim = torch.exp(torch.sum(sentence_embs * positive_embs, dim=-1) / self.temperature)
        neg_sim = torch.exp(torch.matmul(sentence_embs, negative_embs.T) / self.temperature)
        mom_sim = torch.exp(torch.matmul(sentence_embs, momentum_embs.T) / self.temperature)
        neg_sim = torch.cat([neg_sim, mom_sim], dim=1)
  
        # Compute denominator  
        denom = pos_sim + torch.sum(neg_sim, dim=1)  

        # Update the momentum encoder parameters  
        self._momentum_update()
  
        # Compute loss  
        loss = -torch.log(pos_sim / denom)  
        loss = loss.mean()  
  
        return (loss, sentence_outputs) if return_outputs else loss     
  
    @torch.no_grad()  
    def _dequeue_and_enqueue(self, keys):  
        keys = concat_all_gather(keys)  
        batch_size = keys.shape[0]  
    
        ptr = int(self.queue_ptr)  
    
        # Update the queue  
        if ptr + batch_size <= self.queue_size:  
            self.queue[ptr:ptr + batch_size, :] = keys  
        else:  
            overflow = ptr + batch_size - self.queue_size  
            self.queue[ptr:, :] = keys[:batch_size - overflow, :]  
            self.queue[:overflow, :] = keys[batch_size - overflow:, :]  
    
        ptr = (ptr + batch_size) % self.queue_size  # Move pointer  
    
        self.queue_ptr[0] = ptr  # Update pointer    
    
    @torch.no_grad()    
    def _momentum_update(self):    
        for param_q, param_k in zip(self.model.parameters(), self.momentum_encoder.parameters()):  
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

  
@torch.no_grad()    
def concat_all_gather(tensor):    
    if torch.distributed.is_available() and torch.distributed.is_initialized():    
        tensors_gather = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]    
        torch.distributed.all_gather(tensors_gather, tensor)    
        output = torch.cat(tensors_gather, dim=0)    
        return output    
    else:    
        return tensor    
  
def main():  
    from arguments import build_args  
  
    model_args, data_args, training_args = build_args()  
  
    set_seed(training_args.seed)  
    set_logging()  
    logger.info(  
        f"*** Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"  
        + f" distributed training: {bool(training_args.local_rank != -1)}, fp16 training: {training_args.fp16}"  
    )  
    logger.info(f"*** Model parameters {model_args}")  
    logger.info(f"*** Data parameters {data_args}")  
    logger.info(f"*** Training/evaluation parameters {training_args}")  
  
    # 加载 tokenizer  
    tokenizer = RobertaTokenizer.from_pretrained(  
        '../pretrained_models/graphcodebert-base',  
        cache_dir=os.environ.get('TRANSFORMERS_CACHE'),  # 使用自定义缓存目录
        trust_remote_code=True,  
    )  
    if tokenizer.pad_token is None:  
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.pad_token_id = tokenizer.eos_token_id   
        logger.info(f"*** set tokenizer.pad_token to tokenizer.eos_token {tokenizer.eos_token}")  
    logger.info(f"*** tokenizer.padding_side: {tokenizer.padding_side}")  
  
    # 加载数据集  
    train_dataset = None  
    eval_dataset = None  
  
    with training_args.main_process_first(desc="build sft_dataset"):  
        train_dataset = build_dataset(  
            data_args.dataset_name_or_path,  
            "train",  
            tokenizer,  
            num_proc=data_args.preprocessing_num_workers,  
            block_size=data_args.block_size,  
            load_from_cache_file=False,  
        )  
        n_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)  
        logger.info(f"*** builded train samples, total: {n_train_samples}")  
        logger.info(f"*** train_dataset example: {train_dataset[1]}")  
  
    if training_args.do_eval:  
        with training_args.main_process_first(desc="build eval sft_dataset"):  
            eval_dataset = build_dataset(  
                data_args.eval_dataset_name_or_path,  
                "train",  
                tokenizer,  
                num_proc=data_args.preprocessing_num_workers,  
                block_size=data_args.block_size,  
                load_from_cache_file=False,  
            )  
            logger.info(f"*** builded eval samples, total: {len(eval_dataset)}")  
            logger.info(f"*** eval_dataset example: {eval_dataset[1]}")  
  
    logger.info("*** Loading pretrained model for RobertaModel")  
    torch_dtype = model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)  
    logger.info(f"*** torch_dtype {str(torch_dtype)}")  
  
    model = RobertaModel.from_pretrained(  
        '../pretrained_models/graphcodebert-base',  
        torch_dtype=torch_dtype,  
        use_cache=False if training_args.gradient_checkpointing else True,  
        trust_remote_code=True,  
        output_hidden_states=True,  
    )  

    saved_state_dict = torch.load(model_args.model_name_or_path, map_location=model.device)
    modified_state_dict = {k[len('encoder.'):] if k.startswith('encoder.') else k: v for k, v in saved_state_dict.items()}   
    model_state_dict = model.state_dict()    
    model_state_dict.update(modified_state_dict)   
    model.load_state_dict(model_state_dict, strict=False)  
  
    logger.info("*** Loaded pretrained model to CPU")    
    # 定义 collate_fn  
    IGNORE_INDEX = -100  
  
    def collate_fn(batch):  
        result = {}  
        # 准备 sentence batch  
        sentence_batch_dict = {'input_ids': [d["sentence"][0] for d in batch]}     
        sentence_batch_dict = tokenizer.pad(sentence_batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')  
  
        # 准备 positive batch  
        positives_batch_dict = {'input_ids': [d["positive"][0] for d in batch]}  
        positives_batch_dict = tokenizer.pad(positives_batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')  
  
        # 准备 negative batch  
        negative_batch_dict = {'input_ids': [d["negative"][0] for d in batch]}  
        negative_batch_dict = tokenizer.pad(negative_batch_dict, padding=True, return_attention_mask=True, return_tensors='pt')  
  
        # 组合所有部分到 result  
        for k, v in sentence_batch_dict.items():  
            result[f"sentence_{k}"] = v  
  
        for k, v in positives_batch_dict.items():  
            result[f"positive_{k}"] = v  
  
        for k, v in negative_batch_dict.items():  
            result[f"negative_{k}"] = v  
  
        # 确保 labels 是与输入张量相同设备上的张量  
        device = sentence_batch_dict['input_ids'].device  
        result["labels"] = torch.zeros(len(batch), dtype=torch.long, device=device)  
        return result  
  
    # 初始化 Trainer  
    trainer = MyTrainer(  
        model=model,  
        args=training_args,  
        train_dataset=train_dataset,   
        eval_dataset=eval_dataset if training_args.do_eval else None,  
        tokenizer=tokenizer,  
        data_collator=collate_fn,  
    )  
  
    # 添加 MomentumEncoderCallback  
    trainer.add_callback(MomentumEncoderCallback(trainer.momentum_encoder, trainer.model, momentum=0.995))  
  
    # 训练循环  
    if training_args.do_train:  
        logger.info("*** Train")  
        checkpoint = None  
        last_checkpoint = None  
        if training_args.resume_from_checkpoint is not None:  
            checkpoint = training_args.resume_from_checkpoint  
        elif last_checkpoint is not None:  
            checkpoint = last_checkpoint  
        train_result = trainer.train(resume_from_checkpoint=checkpoint)  
  
        metrics = train_result.metrics  
        metrics["train_samples"] = n_train_samples  
  
        trainer.log_metrics("train", metrics)  
        trainer.save_metrics("train", metrics)  
        trainer.save_model(training_args.output_dir)  
  
if __name__ == "__main__":  
    main()  
