# batch size 12 for 16 GB GPU

# 用于code detection finetuning的启动命令
CUDA_VISIBLE_DEVICES=2,3  nohup torchrun \
    --nproc_per_node=2 ./main.py \
    --dataset_name "CR"\
    --train_epochs 30 \
    --model_type "t5cr" \
    --max_source_length 512 \
    --max_target_length 128 \
    --train_batch_size 22 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 3 \
    --mask_rate 0.15 \
    --save_steps 3600 \
    --log_steps 100 \
    --train_steps 120000 \
    --gpu_per_node=2 \
    --node_index=0 \
    --seed 2233 > Output/Log/t5cr_cls_finetune.log 2>&1 & disown



CUDA_VISIBLE_DEVICES=3  nohup torchrun \
    --nproc_per_node=1 ./main.py \
    --dataset_name "CR"\
    --train_epochs 30 \
    --train_eval \
    --model_type "t5cr" \
    --max_source_length 512 \
    --max_target_length 128 \
    --train_batch_size 22 \
    --learning_rate 3e-4 \
    --gradient_accumulation_steps 3 \
    --mask_rate 0.15 \
    --save_steps 3600 \
    --log_steps 100 \
    --train_steps 120000 \
    --gpu_per_node=1 \
    --node_index=0 \
    --load_model_path "../ACR_Model_Saved/t5cr/cls/checkpoint-28800-0.6907/"\
    --seed 2233 > Output/Log/t5cr_cls_test.log 2>&1 & disown