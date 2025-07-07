#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodelist=bharatgpt010,bharatgpt012,bharatgpt155,bharatgpt157,bharatgpt158
#SBATCH --job-name=grvqa_swift_training
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=64
#SBATCH --error=error_log_%j.txt
#SBATCH --output=output_log_%j.txt



# Kill the dangling containers of previous job if they still exist 
CONTAINER_NAME="amruth_ddp_train_swifty"
srun --nodes=${SLURM_NNODES} --ntasks=${SLURM_NNODES} sg docker -c $'
if docker ps -a --format "{{.Names}}" | grep -wq "${CONTAINER_NAME}"; then
	hostname
	echo ":  Container ${CONTAINER_NAME} is being removed ..."
	docker rm -f ${CONTAINER_NAME}
else
	echo "No such contianer found: ${CONTAINER_NAME}"
fi
'
