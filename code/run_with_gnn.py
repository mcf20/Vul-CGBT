import argparse  
import glob  
import logging  
import os  
import pickle  
import random  
import re  
import shutil  
  
import numpy as np  
import torch  
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset  
from torch.utils.data.distributed import DistributedSampler  
import json  
try:  
    from torch.utils.tensorboard import SummaryWriter  
except:  
    from tensorboardX import SummaryWriter  
  
from tqdm import tqdm, trange  
import multiprocessing  
from model import Model, Modelwithcfgdfg  
cpu_cont = multiprocessing.cpu_count()  
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,  
                          BertConfig, BertModel, BertForMaskedLM, BertTokenizer, BertForSequenceClassification,  
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,  
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,  
                          RobertaConfig, RobertaModel, RobertaForSequenceClassification, RobertaTokenizer,  
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertForSequenceClassification, DistilBertTokenizer)  
import torch  
import torch.nn.functional as F  
from torch_geometric.nn import GCNConv  
from torch_geometric.data import Data  
from gensim.models import KeyedVectors  
from gcn_model import GCN, text_to_embedding, build_cfg_data_list, build_dfg_data_list  
from sklearn.metrics import precision_score, recall_score, f1_score  
from torch_geometric.data import Batch  
  
logger = logging.getLogger(__name__)  
  
MODEL_CLASSES = {  
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),  
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),  
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),  
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),  
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)  
}  
  
from gensim.models import KeyedVectors  
import numpy as np  
  
# Load Google News pre-trained model  
model_path = '../data/GoogleNews-vectors-negative300.bin.gz'   
word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)  
  
  
class InputFeatures(object):  
    """A single training/test features for a example."""  
    def __init__(self,  
                 input_tokens,  
                 input_ids,  
                 idx,  
                 label,  
                 cfg_nodes,  
                 cfg_edges,  
                 dfg_nodes,  
                 dfg_edges,  
    ):  
        self.input_tokens = input_tokens  
        self.input_ids = input_ids  
        self.idx = str(idx)  
        self.label = label  
        self.cfg_nodes = cfg_nodes  
        self.cfg_edges = cfg_edges  
        self.dfg_nodes = dfg_nodes  
        self.dfg_edges = dfg_edges  
  
  
def convert_examples_to_features(js, tokenizer, args):  
    # source  
    code = ' '.join(js['func'].split())  
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]  
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]  
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)  
    padding_length = args.block_size - len(source_ids)  
    source_ids += [tokenizer.pad_token_id]*padding_length  
    cfg_nodes = [node[1] for node in js['cfg_nodes']]  
    if len(cfg_nodes) == 0:  
        cfg_nodes = torch.zeros((1, word2vec_model.vector_size), dtype=torch.long)  
    else:  
        cfg_nodes = torch.stack([text_to_embedding(text, word2vec_model) for text in cfg_nodes])  
  
    cfg_edges = js['cfg_edges']  
    if len(cfg_edges) == 0:  
        cfg_edges = torch.zeros((2, 0), dtype=torch.long)  
    else:  
        cfg_edges = torch.tensor(cfg_edges, dtype=torch.long).t().contiguous()  
  
    dfg_nodes = [node[1] for node in js['dfg_nodes']]  
    if len(dfg_nodes) == 0:  
        dfg_nodes = torch.zeros((1, word2vec_model.vector_size), dtype=torch.long)  
    else:  
        dfg_nodes = torch.stack([text_to_embedding(text, word2vec_model) for text in dfg_nodes])  
  
    dfg_edges = js['dfg_edges']  
    if len(dfg_edges) == 0:  
        dfg_edges = torch.zeros((2, 0), dtype=torch.long)  
    else:  
        dfg_edges = torch.tensor(dfg_edges, dtype=torch.long).t().contiguous()  
    return InputFeatures(source_tokens, source_ids, js['idx'], js['target'], cfg_nodes, cfg_edges, dfg_nodes, dfg_edges)  
  
  
class TextDataset(Dataset):  
    def __init__(self, tokenizer, args, file_path=None):  
        self.examples = []  
        with open(file_path) as f:  
            for line in f:  
                js = json.loads(line.strip())  
                self.examples.append(convert_examples_to_features(js, tokenizer, args))  
  
    def __len__(self):  
        return len(self.examples)  
  
    def __getitem__(self, i):  
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), self.examples[i].cfg_nodes, self.examples[i].cfg_edges, self.examples[i].dfg_nodes, self.examples[i].dfg_edges  
  
  
def set_seed(seed=42):  
    random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True  
  
  
def custom_collate_fn(batch):  
    input_ids_list = []  
    labels_list = []  
    cfg_nodes_list = []  
    cfg_edges_list = []  
    dfg_nodes_list = []  
    dfg_edges_list = []  
  
    for item in batch:  
        input_ids_list.append(item[0])  
        labels_list.append(item[1])  
        cfg_nodes_list.append(item[2])  
        cfg_edges_list.append(item[3])  
        dfg_nodes_list.append(item[4])  
        dfg_edges_list.append(item[5])  
  
    input_ids_batch = torch.stack(input_ids_list)  
    labels_batch = torch.stack(labels_list)  
  
    return input_ids_batch, labels_batch, cfg_nodes_list, cfg_edges_list, dfg_nodes_list, dfg_edges_list  
  
  
def train(args, train_dataset, model, tokenizer):  
    """ Train the model """  
    args.train_batch_size = args.per_gpu_train_batch_size  
    if args.local_rank == -1:  
        train_sampler = RandomSampler(train_dataset)  
    else:  
        train_sampler = DistributedSampler(train_dataset)  
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,  
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=False, collate_fn=custom_collate_fn)  
    args.max_steps = args.epoch * len(train_dataloader)  
    args.save_steps = len(train_dataloader)  
    args.warmup_steps = len(train_dataloader)  
    args.logging_steps = len(train_dataloader)  
    args.num_train_epochs = args.epoch  
    model.to(args.device)  
    # Prepare optimizer and schedule (linear warmup and decay)  
    no_decay = ['bias', 'LayerNorm.weight']  
    optimizer_grouped_parameters = [  
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],  
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},  
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.learning_rate}  
    ]  
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)  
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.max_steps*0.1),  
                                                num_training_steps=args.max_steps)  
    if args.fp16:  
        try:  
            from apex import amp  
        except ImportError:  
            raise ImportError("Please install apex to use fp16 training.")  
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)  
  
    if args.local_rank != -1:  
        model = torch.nn.parallel.DistributedDataParallel(  
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True  
        )  
    elif args.n_gpu > 1:  
        model = torch.nn.DataParallel(model)  
  
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')  
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')  
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')  
    if os.path.exists(scheduler_last):  
        scheduler.load_state_dict(torch.load(scheduler_last))  
    if os.path.exists(optimizer_last):  
        optimizer.load_state_dict(torch.load(optimizer_last))  
    # Train!  
    if args.local_rank in [-1, 0]:  
        logger.info("***** Running training *****")  
        logger.info("  Num examples = %d", len(train_dataset))  
        logger.info("  Num Epochs = %d", args.num_train_epochs)  
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)  
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",  
                    args.train_batch_size * args.gradient_accumulation_steps * (  
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))  
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)  
        logger.info("  Total optimization steps = %d", args.max_steps)  
  
    global_step = args.start_step  
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0  
    best_acc, best_f1 = 0.0, 0.0  
    model.zero_grad()  
  
    # Initialize early stopping parameters at the start of training  
    early_stopping_counter = 0  
    best_loss = None  
  
    for idx in range(args.start_epoch, int(args.num_train_epochs)):  
        if args.local_rank != -1:  
            train_sampler.set_epoch(idx)  
        bar = tqdm(train_dataloader, total=len(train_dataloader), disable=args.local_rank not in [-1, 0])  
        tr_num = 0  
        train_loss = 0  
        for step, batch in enumerate(bar):  
            inputs = batch[0].to(args.device)  
            labels = batch[1].to(args.device)  
            cfg_nodes, cfg_edges, dfg_nodes, dfg_edges = batch[2], batch[3], batch[4], batch[5]  
            cfg_batch = build_cfg_data_list(cfg_nodes, cfg_edges).to(args.device)  
            dfg_batch = build_dfg_data_list(dfg_nodes, dfg_edges).to(args.device)  
            model.train()  
            loss, logits = model(inputs, labels, cfg_batch, dfg_batch)  
  
            if args.n_gpu > 1:  
                loss = loss.mean()  
  
            if args.gradient_accumulation_steps > 1:  
                loss = loss / args.gradient_accumulation_steps  
  
            if args.fp16:  
                with amp.scale_loss(loss, optimizer) as scaled_loss:  
                    scaled_loss.backward()  
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)  
            else:  
                loss.backward()  
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  
  
            tr_loss += loss.item()  
            tr_num += 1  
            train_loss += loss.item()  
            avg_loss = round(train_loss / tr_num, 5)  
            if args.local_rank in [-1, 0]:  
                bar.set_description("epoch {} loss {}".format(idx, avg_loss))  
  
            if (step + 1) % args.gradient_accumulation_steps == 0:  
                optimizer.step()  
                optimizer.zero_grad()  
                scheduler.step()  
                global_step += 1  
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)  
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:  
                    logging_loss = tr_loss  
                    tr_nb = global_step  
  
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    if args.evaluate_during_training:  
                        # Synchronize all processes before evaluation  
                        if args.local_rank != -1:  
                            torch.distributed.barrier()  
  
                        # All processes call evaluate  
                        results = evaluate(args, model, tokenizer, eval_when_training=True)  
  
                        # Synchronize all processes after evaluation  
                        if args.local_rank != -1:  
                            torch.distributed.barrier()  
  
                        # Only the main process logs and saves  
                        if args.local_rank in [-1, 0]:  
                            for key, value in results.items():  
                                logger.info("  %s = %s", key, round(value, 4))  
                            if "reveal" in args.eval_data_file or "bigvul" in args.eval_data_file:
                                if results['eval_f1'] > best_f1:  
                                    best_f1 = results['eval_f1']  
                                    logger.info("  " + "*" * 20)  
                                    logger.info("  Best F1:%s", round(best_f1, 4))  
                                    logger.info("  " + "*" * 20)  
    
                                    checkpoint_prefix = 'checkpoint-best-acc'  
                                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                                    if not os.path.exists(output_dir):  
                                        os.makedirs(output_dir)  
                                    model_to_save = model.module if hasattr(model, 'module') else model  
                                    output_model_file = os.path.join(output_dir, 'model.bin')  
                                    torch.save(model_to_save.state_dict(), output_model_file)  
                                    logger.info("Saving model checkpoint to %s", output_dir)
                            else:
                                if results['eval_acc'] > best_acc:  
                                    best_acc = results['eval_acc']  
                                    logger.info("  " + "*" * 20)  
                                    logger.info("  Best acc:%s", round(best_acc, 4))  
                                    logger.info("  " + "*" * 20)  
    
                                    checkpoint_prefix = 'checkpoint-best-acc'  
                                    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
                                    if not os.path.exists(output_dir):  
                                        os.makedirs(output_dir)  
                                    model_to_save = model.module if hasattr(model, 'module') else model  
                                    output_model_file = os.path.join(output_dir, 'model.bin')  
                                    torch.save(model_to_save.state_dict(), output_model_file)  
                                    logger.info("Saving model checkpoint to %s", output_dir)  
  
        # Early Stopping Check  
        avg_loss_epoch = train_loss / tr_num  
        if args.early_stopping_patience is not None:  
            if best_loss is None or avg_loss_epoch < best_loss - args.min_loss_delta:  
                best_loss = avg_loss_epoch  
                early_stopping_counter = 0  
            else:  
                early_stopping_counter += 1  
                if early_stopping_counter >= args.early_stopping_patience:  
                    logger.info("Early stopping")  
                    break  
  
  
def evaluate(args, model, tokenizer, eval_when_training=False):  
    eval_output_dir = args.output_dir  
  
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)  
  
    if args.local_rank == -1:  
        eval_sampler = SequentialSampler(eval_dataset)  
    else:  
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)  
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)  
  
    # Remove this barrier since all processes are calling evaluate  
    # if args.local_rank != -1:  
    #     torch.distributed.barrier()  
  
    # Do not wrap the model with DataParallel during evaluation  
    # multi-gpu evaluate  
    # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):  
    #     model = torch.nn.DataParallel(model)  
  
    # Eval!  
    logger.info("***** Running evaluation *****")  
    logger.info("  Num examples = %d", len(eval_dataset))  
    logger.info("  Batch size = %d", args.eval_batch_size)  
    eval_loss = torch.tensor(0.0).to(args.device)  
    nb_eval_steps = 0  
    model.eval()  
    logits_list = []  
    labels_list = []  
    for batch in eval_dataloader:  
        inputs = batch[0].to(args.device)  
        label = batch[1].to(args.device)  
        cfg_nodes, cfg_edges, dfg_nodes, dfg_edges = batch[2], batch[3], batch[4], batch[5]  
        cfg_batch = build_cfg_data_list(cfg_nodes, cfg_edges).to(args.device)  
        dfg_batch = build_dfg_data_list(dfg_nodes, dfg_edges).to(args.device)  
        with torch.no_grad():  
            lm_loss, logit = model(inputs, label, cfg_batch, dfg_batch)  
            eval_loss += lm_loss.sum()  
            logits_list.append(logit)  
            labels_list.append(label)  
        nb_eval_steps += 1  
  
    # Gather eval_loss from all processes  
    if args.local_rank != -1:  
        torch.distributed.all_reduce(eval_loss, op=torch.distributed.ReduceOp.SUM)  
        eval_loss = eval_loss.item() / torch.distributed.get_world_size()  
    else:  
        eval_loss = eval_loss.item()  
  
    # Convert lists to tensors  
    logits = torch.cat(logits_list, dim=0)  
    labels = torch.cat(labels_list, dim=0)  
  
    # Gather logits and labels from all processes  
    if args.local_rank != -1:  
        # Determine the total number of samples across all processes  
        total_samples = torch.tensor([logits.size(0)], dtype=torch.long).to(args.device)  
        torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)  
        total_samples = total_samples.item()  
  
        gathered_logits = [torch.zeros_like(logits) for _ in range(torch.distributed.get_world_size())]  
        gathered_labels = [torch.zeros_like(labels) for _ in range(torch.distributed.get_world_size())]  
  
        torch.distributed.all_gather(gathered_logits, logits)  
        torch.distributed.all_gather(gathered_labels, labels)  
  
        all_logits = torch.cat(gathered_logits, dim=0)[:total_samples]  
        all_labels = torch.cat(gathered_labels, dim=0)[:total_samples]  
    else:  
        all_logits = logits  
        all_labels = labels  
  
    # Proceed to compute metrics  
    if args.local_rank in [-1, 0]:  
        logits_np = all_logits.cpu().numpy()  
        labels_np = all_labels.cpu().numpy()  
        preds = logits_np[:, 0] > 0.5  
        eval_acc = np.mean(labels_np == preds)  
  
        preds = preds.astype(int)  
        f1 = f1_score(labels_np.tolist(), preds.tolist(), average='binary')  
  
        result = {  
            "eval_loss": float(eval_loss),  
            "eval_acc": round(eval_acc, 4),  
            "eval_f1": round(f1, 4),  
        }  
  
        logger.info("***** Eval results *****")  
        for key in sorted(result.keys()):  
            logger.info("  %s = %s", key, str(result[key]))  
  
        return result  
    else:  
        return None  # Non-zero ranks return None  
  
  
def test(args, model, tokenizer):  
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)  
  
    if args.local_rank == -1:  
        eval_sampler = SequentialSampler(eval_dataset)  
    else:  
        eval_sampler = DistributedSampler(eval_dataset, shuffle=False)  
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)  
  
    # Do not wrap the model with DataParallel during evaluation  
    # if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):  
    #     model = torch.nn.DataParallel(model)  
  
    # Eval!  
    logger.info("***** Running Test *****")  
    logger.info("  Num examples = %d", len(eval_dataset))  
    logger.info("  Batch size = %d", args.eval_batch_size)  
    model.eval()  
    logits_list = []  
    labels_list = []  
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):  
        inputs = batch[0].to(args.device)  
        label = batch[1].to(args.device)  
        cfg_nodes, cfg_edges, dfg_nodes, dfg_edges = batch[2], batch[3], batch[4], batch[5]  
        cfg_batch = build_cfg_data_list(cfg_nodes, cfg_edges).to(args.device)  
        dfg_batch = build_dfg_data_list(dfg_nodes, dfg_edges).to(args.device)  
        with torch.no_grad():  
            logit = model(inputs, label=None, cfg_batch=cfg_batch, dfg_batch=dfg_batch)  
            logits_list.append(logit)  
            labels_list.append(label)  
  
    # Gather logits and labels from all processes  
    if args.local_rank != -1:  
        logits = torch.cat(logits_list, dim=0)  
        labels = torch.cat(labels_list, dim=0)  
  
        total_samples = torch.tensor([logits.size(0)], dtype=torch.long).to(args.device)  
        torch.distributed.all_reduce(total_samples, op=torch.distributed.ReduceOp.SUM)  
        total_samples = total_samples.item()  
  
        gathered_logits = [torch.zeros_like(logits) for _ in range(torch.distributed.get_world_size())]  
        gathered_labels = [torch.zeros_like(labels) for _ in range(torch.distributed.get_world_size())]  
  
        torch.distributed.all_gather(gathered_logits, logits)  
        torch.distributed.all_gather(gathered_labels, labels)  
  
        all_logits = torch.cat(gathered_logits, dim=0)[:total_samples].cpu().numpy()  
        all_labels = torch.cat(gathered_labels, dim=0)[:total_samples].cpu().numpy()  
    else:  
        all_logits = torch.cat(logits_list, dim=0).cpu().numpy()  
        all_labels = torch.cat(labels_list, dim=0).cpu().numpy()  
  
    preds = all_logits[:, 0] > 0.5  
    if args.local_rank in [-1, 0]:  
        with open(os.path.join(args.output_dir, "predictions.txt"), 'w') as f:  
            for example, pred in zip(eval_dataset.examples, preds):  
                if pred:  
                    f.write(example.idx + '\t1\n')  
                else:  
                    f.write(example.idx + '\t0\n')  
  
  
def main():  
    parser = argparse.ArgumentParser()  
  
    ## Required parameters  
    parser.add_argument("--train_data_file", default=None, type=str, required=True,  
                        help="The input training data file (a text file).")  
    parser.add_argument("--output_dir", default=None, type=str, required=True,  
                        help="The output directory where the model predictions and checkpoints will be written.")  
  
    ## Other parameters  
    parser.add_argument("--eval_data_file", default=None, type=str,  
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")  
    parser.add_argument("--test_data_file", default=None, type=str,  
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")  
  
    parser.add_argument("--model_type", default="bert", type=str,  
                        help="The model architecture to be fine-tuned.")  
    parser.add_argument("--model_name_or_path", default=None, type=str,  
                        help="The model checkpoint for weights initialization.")  
    parser.add_argument("--saved_model_bin_path", default=None, type=str,  
                        help="The model checkpoint for weights initialization.")  
  
    parser.add_argument("--mlm", action='store_true',  
                        help="Train with masked-language modeling loss instead of language modeling.")  
    parser.add_argument("--mlm_probability", type=float, default=0.15,  
                        help="Ratio of tokens to mask for masked language modeling loss")  
  
    parser.add_argument("--config_name", default="", type=str,  
                        help="Optional pretrained config name or path if not the same as model_name_or_path")  
    parser.add_argument("--tokenizer_name", default="", type=str,  
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")  
    parser.add_argument("--cache_dir", default="", type=str,  
                        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)")  
    parser.add_argument("--block_size", default=-1, type=int,  
                        help="Optional input sequence length after tokenization."  
                             "The training dataset will be truncated in block of this size for training."  
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")  
    parser.add_argument("--do_train", action='store_true',  
                        help="Whether to run training.")  
    parser.add_argument("--do_eval", action='store_true',  
                        help="Whether to run eval on the dev set.")  
    parser.add_argument("--do_test", action='store_true',  
                        help="Whether to run eval on the test set.")  
    parser.add_argument("--evaluate_during_training", action='store_true',  
                        help="Run evaluation during training at each logging step.")  
    parser.add_argument("--do_lower_case", action='store_true',  
                        help="Set this flag if you are using an uncased model.")  
  
    parser.add_argument("--train_batch_size", default=4, type=int,  
                        help="Batch size per GPU/CPU for training.")  
    parser.add_argument("--eval_batch_size", default=4, type=int,  
                        help="Batch size per GPU/CPU for evaluation.")  
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,  
                        help="Number of updates steps to accumulate before performing a backward/update pass.")  
    parser.add_argument("--learning_rate", default=5e-5, type=float,  
                        help="The initial learning rate for Adam.")  
    parser.add_argument("--weight_decay", default=0.0, type=float,  
                        help="Weight decay if we apply some.")  
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,  
                        help="Epsilon for Adam optimizer.")  
    parser.add_argument("--max_grad_norm", default=1.0, type=float,  
                        help="Max gradient norm.")  
    parser.add_argument("--epoch", type=int, default=1,  
                        help="Total number of training epochs to perform.")  
    parser.add_argument("--max_steps", default=-1, type=int,  
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")  
    parser.add_argument("--warmup_steps", default=0, type=int,  
                        help="Linear warmup over warmup_steps.")  
  
    parser.add_argument('--logging_steps', type=int, default=50,  
                        help="Log every X updates steps.")  
    parser.add_argument('--save_steps', type=int, default=50,  
                        help="Save checkpoint every X updates steps.")  
    parser.add_argument('--save_total_limit', type=int, default=2,  
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')  
    parser.add_argument("--eval_all_checkpoints", action='store_true',  
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")  
    parser.add_argument("--no_cuda", action='store_true',  
                        help="Avoid using CUDA when available")  
    parser.add_argument('--overwrite_output_dir', action='store_true',  
                        help="Overwrite the content of the output directory")  
    parser.add_argument('--overwrite_cache', action='store_true',  
                        help="Overwrite the cached training and evaluation sets")  
    parser.add_argument('--seed', type=int, default=42,  
                        help="random seed for initialization")  
    parser.add_argument('--fp16', action='store_true',  
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")  
    parser.add_argument('--fp16_opt_level', type=str, default='O1',  
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."  
                             "See details at https://nvidia.github.io/apex/amp.html")  
    # Remove the argument parsing for --local_rank  
    # parser.add_argument("--local_rank", "--local-rank", type=int, default=-1,  
    #                     help="For distributed training: local_rank")  
  
    # Add early stopping parameters and dropout probability parameters  
    parser.add_argument("--early_stopping_patience", type=int, default=None,  
                        help="Number of epochs with no improvement after which training will be stopped.")  
    parser.add_argument("--min_loss_delta", type=float, default=0.001,  
                        help="Minimum change in the loss required to qualify as an improvement.")  
    parser.add_argument('--dropout_probability', type=float, default=0.3, help='dropout probability')  

    # cfg dfg abalation study
    parser.add_argument("--only_cfg", action='store_true',  
                    help="only add cfg embedding.")  
    parser.add_argument("--only_dfg", action='store_true',  
                help="only add dfg embedding.")  

    #gnn model pooling abalation study ["mean", "max", "joint"]
    parser.add_argument("--pooling_type", default="mean", type=str,  
                    help="gnn model pooling type")  
  
    args = parser.parse_args()  
  
    # Set up local_rank from environment variable  
    import os  
    if "LOCAL_RANK" in os.environ:  
        args.local_rank = int(os.environ["LOCAL_RANK"])  
    else:  
        args.local_rank = -1  
  
    # Setup CUDA, GPU & distributed training  
    if args.local_rank == -1:  
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")  
        args.n_gpu = torch.cuda.device_count()  
    else:  
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs  
        torch.cuda.set_device(args.local_rank)  
        device = torch.device("cuda", args.local_rank)  
        torch.distributed.init_process_group(backend='nccl')  
        args.n_gpu = 1  
    args.device = device  
    args.per_gpu_train_batch_size = args.train_batch_size  
    args.per_gpu_eval_batch_size = args.eval_batch_size  
  
    # Setup logging  
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',  
                        datefmt='%m/%d/%Y %H:%M:%S',  
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)  
    if args.local_rank in [-1, 0]:  
        logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",  
                       args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)  
  
    # Set seed  
    set_seed(args.seed)  
  
    # Load pretrained model and tokenizer  
    if args.local_rank not in [-1, 0]:  
        torch.distributed.barrier()  
  
    args.start_epoch = 0  
    args.start_step = 0  
  
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]  
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,  
                                          cache_dir=args.cache_dir if args.cache_dir else None)  
    config.num_labels = 1  
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,  
                                                do_lower_case=args.do_lower_case,  
                                                cache_dir=args.cache_dir if args.cache_dir else None)  
    if args.block_size <= 0:  
        args.block_size = tokenizer.model_max_length  
    args.block_size = min(args.block_size, tokenizer.model_max_length)  
    if args.model_name_or_path:  
        model = model_class.from_pretrained(args.model_name_or_path,  
                                            from_tf=bool('.ckpt' in args.model_name_or_path),  
                                            config=config,  
                                            cache_dir=args.cache_dir if args.cache_dir else None)  
    else:  
        model = model_class(config)  
  
    model = Modelwithcfgdfg(model, config, tokenizer, args)  
    if args.saved_model_bin_path:  
        saved_state_dict = torch.load(args.saved_model_bin_path, map_location=args.device)  
        current_state_dict = model.state_dict()  
        matched_state_dict = {k: v for k, v in saved_state_dict.items() if k in current_state_dict and k not in ['classifier.weight', 'classifier.bias']}  
    
        # Load the matched state dict into the model  
        current_state_dict.update(matched_state_dict)  
        model.load_state_dict(current_state_dict, strict=False)
  
    if args.local_rank == 0:  
        torch.distributed.barrier()  
  
    if args.local_rank in [-1, 0]:  
        logger.info("Training/evaluation parameters %s", args)  
  
    # Training  
    if args.do_train:  
        if args.local_rank not in [-1, 0]:  
            torch.distributed.barrier()  
  
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)  
        if args.local_rank == 0:  
            torch.distributed.barrier()  
  
        train(args, train_dataset, model, tokenizer)  
  
    # Evaluation  
    if args.do_eval:  
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'  
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)  
        model.to(args.device)  
        evaluate(args, model, tokenizer)  
  
    if args.do_test:  
        checkpoint_prefix = 'checkpoint-best-acc/model.bin'  
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)  
        model.to(args.device)  
        test(args, model, tokenizer)  
  
  
if __name__ == "__main__":  
    main()  
