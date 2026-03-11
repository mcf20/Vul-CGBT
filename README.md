# Vul-CGBT
Vul-CGBT: Enhancing Code Vulnerability Detection via Contrastive Semantic Learning and Graph Embedding
# Vulnerability Detection
## Task Define
Given a source code, the task is to identify whether it is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack.  We treat the task as binary classification (0/1), where 1 stands for insecure code and 0 for secure code.

## environment install

1. Change to the root directory.  #cd 
2.You need to have Anaconda installed beforehand.
3.run

```shell
conda env create -f environment.yaml  
```
4.activate environment
```shell
conda activate environment
```

### Finetune
### Vul-CGBT
```shell
cd code
chmod +x batch_run.sh
./batch_run.sh
```

### MLM+MCL
```shell
cd code
chmod +x batch_run_mlm_mcl.sh
./batch_run_mlm_cl.sh
```

### MLM
```shell
cd code
chmod +x batch_run_mlm.sh
./batch_run_mlm.sh
```

### Test
1. For models incorporating GNNs, including those using CFG alone, DFG alone, and both CFG and DFG simultaneously.
Take testing Devign as an example.
```shell
cd code
export CUDA_VISIBLE_DEVICES=0
python run_with_gnn.py \
    --output_dir=../saved_models/mlm_mcl_gnn_on_devign \
    --model_type=roberta \
    --tokenizer_name=../pretrained_models/graphcodebert-base/ \
    --model_name_or_path=../pretrained_models/graphcodebert-base/ \
    --do_eval \
    --train_data_file=../dataset/devign/train.jsonl \
    --eval_data_file=../dataset/devign/test.jsonl \
    --test_data_file=../dataset/devign/test.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 40 \
    --eval_batch_size 40 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 990302 \
    --saved_model_bin_path=../pretrained_models/mlm_cl_gnn/model.bin 
```
2. For models that do not incorporate GNNs (MLM+MCL,MLM)
Take testing Devign as an example.
```shell
cd code
python run.py \
    --output_dir=../saved_models/mlm_mcl_on_devign \
    --model_type=roberta \
    --tokenizer_name=../pretrained_models/graphcodebert-base/ \
    --model_name_or_path=../pretrained_models/graphcodebert-base/ \
    --do_eval \
    --train_data_file=../dataset/devign/train.jsonl \
    --eval_data_file=../dataset/devign/test.jsonl \
    --test_data_file=../dataset/devign/test.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 40 \
    --eval_batch_size 40 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 990302 \
    --saved_model_bin_path=../pretrained_models/mlm_cl/model.bin 
```
## Pretrain task
### MLM
```shell
cd pretrain
torchrun --nproc_per_node=8 mlm_pretrain.py \
--output_dir "MLM" \
--model_name_or_path "../pretrained_models/graphcodebert-base" \
--dataset_name_or_path "MLM pretrain.json" \
--block_size 512 \
--preprocessing_num_workers 20 \
--fp16 true \
--do_train true \
--do_eval false \
--gradient_accumulation_steps 4 \
--gradient_checkpointing true \
--learning_rate 1.0e-05 \
--logging_steps 1 \
--logging_strategy steps \
--lr_scheduler_type cosine \
--max_steps -1 \
--num_train_epochs 10 \
--overwrite_output_dir true \
--per_device_eval_batch_size 1 \
--per_device_train_batch_size 1024 \
--remove_unused_columns true \
--save_steps 300 \
--save_strategy steps \
--save_total_limit 3 \
--seed 24 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
```

### MCL task
```shell
cd pretrain
torchrun --nproc_per_node=8 momentum_cl_pretrain.py \
--output_dir "MCL" \
--model_name_or_path "MLM" \
--dataset_name_or_path "MCL pretrain.json" \
--block_size 512 \
--preprocessing_num_workers 20 \
--fp16 true \
--do_train true \
--do_eval false \
--gradient_accumulation_steps 4 \
--gradient_checkpointing true \
--learning_rate 1.0e-05 \
--logging_steps 1 \
--logging_strategy steps \
--lr_scheduler_type cosine \
--max_steps -1 \
--num_train_epochs 10 \
--overwrite_output_dir true \
--per_device_eval_batch_size 1 \
--per_device_train_batch_size 64 \
--remove_unused_columns true \
--save_steps 300 \
--save_strategy steps \
--save_total_limit 3 \
--seed 24 \
--warmup_ratio 0.1 \
--weight_decay 0.01 \
