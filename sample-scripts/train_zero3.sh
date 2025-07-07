
# Default USERNAME if not set
USERNAME=${USERNAME:-venkat_kesav}

# Activate the virtual environment
source /home/${USERNAME}/.venv/bin/activate

nproc_per_node=2

# Environmental Variables
export CUDA_VISIBLE_DEVICES=4,5
export MODELSCOPE_CACHE="/workspace/models" 
export WANDB_API_KEY="2f5d0df5148bfa9175469270ad15c176dc23dcfd"
export NPROC_PER_NODE=$nproc_per_node 

swift sft \
    --model /workspace/models/Qwen2.5-VL-7B-Instruct \
    --train_type lora \
    --dataset swift/Mantis-Instruct:docvqa#1000 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 2 \
    --model_author swift \
    --model_name swift-robot \
    --report_to wandb \
    --deepspeed zero3
