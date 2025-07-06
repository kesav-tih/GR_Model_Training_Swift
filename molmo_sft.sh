#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodelist=bharatgpt010,bharatgpt012,bharatgpt155,bharatgpt157,bharatgpt158
#SBATCH --job-name=grvqa_swift_training
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --error=error_log_%j.txt
#SBATCH --output=output_log_%j.txt
#SBATCH --mail-user=hrithik.sagar@tihiitb.org
#SBATCH --mail-type=ALL

# Default USERNAME if not set
export USERNAME="venkat_kesav"
export MODELSCOPE_CACHE="/workspace/models" 

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "hostname -I | tr ' ' '\n' | grep '^10\.' | head -n 1")
head_node_port=$((10000 + SLURM_JOBID%10000))

srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} sg docker -c 'hostname && docker load -i ./swift_vis_grp.tar'

export LOGLEVEL=INFO
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export WORLD_SIZE=$(($SLURM_NNODES * 8))
export MASTER_ADDR=${head_node_ip}
export MASTER_PORT=${head_node_port}

# Creating a folder for storing checkpoints, job-wise.
srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "mkdir output_${SLURM_JOBID}"

# Kill the dangling containers of previous job if they still exist
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} sg docker -c 'docker rm -f kesav_ddp_train_swift || true'

# Run new container
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} sg docker -c 'docker run --shm-size=256g --rm --net=host --gpus all \
--name kesav_ddp_train_swift \
-e WORLD_SIZE=${WORLD_SIZE} \
-e RANK=${SLURM_PROCID} \
-e LOCAL_RANK=${SLURM_LOCALID} \
-e NCCL_DEBUG=INFO \
-e NCCL_SOCKET_IFNAME=bond0 \
-e GLOO_SOCKET_IFNAME=bond0 \
-e MODELSCOPE_CACHE=${MODELSCOPE_CACHE} \
-e WANDB_API_KEY="2f5d0df5148bfa9175469270ad15c176dc23dcfd" \
-v /projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift:/workspace \
-v /projects/data/vision-team:/projects/data/vision-team \
-v /projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/output_${SLURM_JOBID}:/workspace/output \
swift:vis_grp bash -c "source /home/venkat_kesav/.venv/bin/activate && \
torchrun \
    --master_port ${MASTER_PORT} \
    --nproc_per_node=8 \
    --nnodes=${SLURM_NNODES} \
    --master_addr=${MASTER_ADDR} \
    --node_rank=${SLURM_NODEID} \
    /home/venkat_kesav/ms-swift/swift/cli/sft.py \
    --model /projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/final_outputs/output_5190/v0-20250526-050001/checkpoint-12440 \
    --train_type full \
    --dataset /projects/data/vision-team/hrithik_sagar/synthetic_qa_outputs/swift_training_format_7 \
    --dataset_num_proc 16 \
    --use_hf True \
    --dataloader_num_workers 16 \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 200 \
    --logging_steps 10 \
    --split_dataset_ratio 0.1 \
    --eval_strategy epoch \
    --max_length 8192 \
    --output_dir /workspace/output \
    --warmup_ratio 0.05 \
    --report_to wandb \
    --model_author ModelScope \
    --model_name molmo-docvqa \
    --max_pixels 4194304 \
    --deepspeed zero3"'    

# Removed the option --save_total_limit ( default is None, but can specify how much to save)
#     --resume_from_checkpoint /projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/output_5115/v0-20250525-063349/checkpoint-6200 \
#     --model allenai/Molmo-7B-O-0924 \
# --max_pixels 16777216 \
# --per_device_train_batch_size 12 \ # Expeirment with 12 later. 