#!/bin/bash  

echo "Starting to run Python scripts..."  
 
echo "Running reveal task"  
python run.py \
    --output_dir=../saved_models/mlm_on_reveal \
    --model_type=roberta \
    --tokenizer_name=../pretrained_models/graphcodebert-base/ \
    --model_name_or_path=../pretrained_models/graphcodebert-base/ \
    --do_train \
    --train_data_file=../dataset/reveal/train_with_cfg_dfg.jsonl \
    --eval_data_file=../dataset/reveal/vale_with_cfg_dfg.jsonl \
    --test_data_file=../dataset/reveal/test_with_cfg_dfg.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 40 \
    --eval_batch_size 40 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 990302 \
    --saved_model_bin_path=../pretrained_models/mlm/model.bin 
  
  
if [ $? -ne 0 ]; then  
    echo "Running reveal task failed"  
    exit 1  
fi  
  

echo "Running devign task"  
python run.py \
    --output_dir=../saved_models/mlm_on_devign \
    --model_type=roberta \
    --tokenizer_name=../pretrained_models/graphcodebert-base/ \
    --model_name_or_path=../pretrained_models/graphcodebert-base/ \
    --do_train \
    --train_data_file=../dataset/devign/train_with_cfg_dfg.jsonl \
    --eval_data_file=../dataset/devign/vale_with_cfg_dfg.jsonl \
    --test_data_file=../dataset/devign/test_with_cfg_dfg.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 40 \
    --eval_batch_size 40 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 990302 \
    --saved_model_bin_path=../pretrained_models/mlm/model.bin 
  
 
if [ $? -ne 0 ]; then  
    echo "Running devign task failed"  
    exit 1  
fi  
  

echo "Running bigvul task"  
python run.py \
    --output_dir=../saved_models/mlm_on_bigvul \
    --model_type=roberta \
    --tokenizer_name=../pretrained_models/graphcodebert-base/ \
    --model_name_or_path=../pretrained_models/graphcodebert-base/ \
    --do_train \
    --train_data_file=../dataset/bigvul/train_with_cfg_dfg.jsonl \
    --eval_data_file=../dataset/bigvul/vale_with_cfg_dfg.jsonl \
    --test_data_file=../dataset/bigvul/test_with_cfg_dfg.jsonl \
    --epoch 10 \
    --block_size 400 \
    --train_batch_size 40 \
    --eval_batch_size 40 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 990302 \
    --saved_model_bin_path=../pretrained_models/mlm/model.bin  
  

if [ $? -ne 0 ]; then  
    echo "Running bigvul task failed"  
    exit 1  
fi  
  
echo "All Python scripts ran successfully."
