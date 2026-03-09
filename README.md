# Vul-CGBT
Vul-CGBT: Enhancing Code Vulnerability Detection via Contrastive Semantic Learning and Graph Embedding
# Vulnerability Detection

## Task Define

Given a source code, the task is to identify whether it is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack.  We treat the task as binary classification (0/1), where 1 stands for insecure code and 0 for secure code.

## environment install

1. 切换到根目录下

2.预先需要安装好Anaconda

3.运行

```shell
conda env create -f environment.yaml  
```
4.激活环境
```shell
conda activate environment
```

### 微调

### MLM+CL+GNN
1. 运行微调的训练脚本
```shell
cd code
chmod +x batch_run.sh
./batch_run.sh
```

### MLM+CL
1. 运行微调的训练脚本
```shell
cd code
chmod +x batch_run_mlm_cl.sh
./batch_run_mlm_cl.sh
```

### MLM
1. 运行微调的训练脚本
```shell
cd code
chmod +x batch_run_mlm.sh
./batch_run_mlm.sh
```

### 测试
测试都需要用单卡来运行
1. 对于包含GNN的模型
以测试Devign举例
```shell
cd code
export CUDA_VISIBLE_DEVICES=0
python run_with_gnn.py \
    --output_dir=../saved_models/mlm_cl_gnn_on_devign \
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

2. 对于不包含GNN的模型
以测试Devign举例
```shell
cd code
python run.py \
    --output_dir=../saved_models/mlm_cl_on_devign \
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
deepspeed \
--include="localhost:0,1,2,3,4,5,6,7" \
mlm_pretrain.py \
--output_dir "选择MLM训练后的模型的存储路径" \
--model_name_or_path "../pretrained_models/graphcodebert-base" \
--dataset_name_or_path "VD MLM pretrain的json文件路径" \
--block_size 512 \
--preprocessing_num_workers 20 \
--deepspeed ./ds_config_zero3.json \
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
deepspeed \
--include="localhost:0,1,2,3,4,5,6,7" \
momentum_cl_pretrain.py \
--output_dir "选择训练后的模型的存储路径" \
--model_name_or_path "选择MLM训练后的模型的存储路径" \
--dataset_name_or_path "VD CL pretrain的json文件路径" \
--block_size 512 \
--preprocessing_num_workers 20 \
--deepspeed ./ds_config_zero3.json \
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
