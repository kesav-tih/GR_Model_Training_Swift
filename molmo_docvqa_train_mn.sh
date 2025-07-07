#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodelist=bharatgpt010,bharatgpt156
#SBATCH --job-name=grvqa_swift_training
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --error=error_log.txt
#SBATCH --output=output_log.txt


# Default USERNAME if not set
export USERNAME="venkat_kesav"
export MODELSCOPE_CACHE="/workspace/models" 


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -c "hostname -I | tr ' ' '\n' | grep '^10\.' | head -n 1")
head_node_port=$((10000 + SLURM_JOBID%10000))

srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} sg docker -c 'docker load -i ./swift.tar'

export LOGLEVEL=INFO
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export WORLD_SIZE=$(($SLURM_NNODES * 8))
export MASTER_ADDR=${head_node_ip}
export MASTER_PORT=${head_node_port}

srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} sg docker -c 'docker run --shm-size=32g --rm --net=host --gpus all \
-v /projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/output:/workspace/output \
-w /workspace \
--name ddp_train_swift \
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
swift_img:tr_and_wb_fix bash -c "source /home/venkat_kesav/.venv/bin/activate && \
torchrun \
    --master_port ${MASTER_PORT} \
    --nproc_per_node=8 \
    --nnodes=${SLURM_NNODES} \
    --master_addr=${MASTER_ADDR} \
    --node_rank=${SLURM_NODEID} \
    /home/venkat_kesav/ms-swift/swift/cli/sft.py \
    --model /workspace/models/Molmo-7B-D-0924 \
    --train_type full \
    --dataset /projects/data/vision-team/hrithik_sagar/synthetic_qa_outputs/swift_training_format_2 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 32 / 8 / ${SLURM_NNODES}) \
    --save_strategy epoch \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --model_author ModelScope \
    --model_name molmo-docvqa \
    --deepspeed zero2"'


