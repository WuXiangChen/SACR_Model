# batch size 12 for 16 GB GPU


# 用于code detection finetuning的启动命令
CUDA_VISIBLE_DEVICES=2,3 nohup torchrun \
    --nproc_per_node=2 ./main.py \
    --dataset_name "CR"\
    --train_epochs 30 \
    --model_type "codereviewer" \
    --max_source_length 512 \
    --max_target_length 128 \
    --train_batch_size 8 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 3 \
    --mask_rate 0.15 \
    --save_steps 3600 \
    --log_steps 100 \
    --train_steps 120000 \
    --gpu_per_node=2 \
    --node_index=0 \
    --seed 2233 > Output/Log/codereviewer_cls_finetune.log 2>&1 & disown

# 用于reviw comments generation finetuning的启动命令
CUDA_VISIBLE_DEVICES=2,3 nohup torchrun \
    --nproc_per_node=2 ./main.py \
    --dataset_name "CR"\
    --train_epochs 30 \
    --model_type "codereviewer" \
    --max_source_length 512 \
    --max_target_length 128 \
    --train_batch_size 8 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 3 \
    --mask_rate 0.15 \
    --save_steps 3600 \
    --log_steps 100 \
    --train_steps 120000 \
    --task_type "msg"\
    --gpu_per_node=2 \
    --node_index=0 \
    --seed 2233 > Output/Log/codereviewer_msg_finetune.log 2>&1 & disown

# 测试命令
CUDA_VISIBLE_DEVICES=2,3 nohup torchrun \
    --nproc_per_node=2 ./main.py \
    --dataset_name "CR"\
    --train_epochs 30 \
    --train_eval \
    --model_type "codereviewer" \
    --max_source_length 512 \
    --max_target_length 128 \
    --train_batch_size 8 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 3 \
    --mask_rate 0.15 \
    --save_steps 3600 \
    --log_steps 100 \
    --train_steps 120000 \
    --task_type "msg"\
    --gpu_per_node=2 \
    --node_index=0 \
    --load_model_path "../ACR_Model_Saved/codereviewer/msg/checkpoint-100-0.0000/"\
    --seed 2233 > Output/Log/codereviewer_msg_test.log 2>&1 & disown
