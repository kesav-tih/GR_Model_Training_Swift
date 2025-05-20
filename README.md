Author: hrithiksagar-tih 
Date: 2025-05-20
Title: Molmo-7B-O-0924 fine-tuning with Swift
Description: This document provides a comprehensive guide on how to run the Molmo-7B-O-0924 fine-tuning job using Swift, including setup, monitoring, and troubleshooting tips.
Tags: [Molmo-7B-O-0924, Swift, fine-tuning, SLURM, Docker, PyTorch, training job]
Version: 1.0
Status: draft
Last_updated: 2025-05-20
Last_updated_by: hrithiksagar-tih

---
# Molmo-7B-O-0924 fine-tuning with Swift
# Table of Contents:
- [Running & monitoring](#running--monitoring)
- [Example](#example)
- [Where do logs and checkpoints go?](#where-do-logs-and-checkpoints-go)
- [1. High-level flow (what happens, in order)](#1-high-level-flow-what-happens-in-order)
- [2. Directory & volume map (host → container)](#2-directory--volume-map-host--container)
- [3. Where files end up](#3-where-files-end-up)
- [4. Line-by-line explanation](#4-line-by-line-explanation)
  - [4.1 SLURM header](#41-slurm-header)
  - [4.2 User-tunable variables](#42-user-tunable-variables)
  - [4.3 Figure out the cluster topology](#43-figure-out-the-cluster-topology)
  - [4.4 Distribute the Docker image](#44-distribute-the-docker-image)
  - [4.5 Set distributed-training env vars](#45-set-distributed-training-env-vars)
  - [4.6 Pre-run housekeeping](#46-pre-run-housekeeping)
  - [4.7 Launch the training job](#47-launch-the-training-job)
- [5. How to **scale to more nodes**](#5-how-to-scale-to-more-nodes)
- [6. Customising paths & image names](#6-customising-paths--image-names)
- [7. Troubleshooting quick hits](#7-troubleshooting-quick-hits)
- [Sleep-mode summary 😴](#sleep-mode-summary-)
---
# Molmo-7B-O-0924 fine-tuning with Swift
## 🚀 How to run the job
### 1. **Login to the cluster**

```bash
module load slurm           # if not auto-loaded on login node
sbatch molmo_sft.sh         # submits the job
# ^ this will start the process, and the person whose emial is giving in the code will get an email when the job starts and ends
# Monitoring the job: 
squeue                      # one-off status
watch squeue                # live view (press q to quit)
tail -f output_log_<jobid>.txt   # stream stdout
tail -f error_log_<jobid>.txt    # stream stderr
```

### Cancelling:

```bash
scancel <jobid>
```
---

## Example

### ✅ **Where do logs and checkpoints go?**

### ✅ 1. **SLURM logs (stdout & stderr)**

These are auto-saved by SLURM when you use:

```bash
#SBATCH --error=error_log_%j.txt
#SBATCH --output=output_log_%j.txt
```

Here:

* `%j` is the SLURM job ID.
* These logs are saved in **the directory where you run `sbatch molmo_sft.sh`**.

---

### 🔍 **Example:**

If your SLURM job ID was `4732`, then you'll find:

| File name             | Meaning                                                   |
| --------------------- | --------------------------------------------------------- |
| `error_log_4732.txt`  | **stderr** output (errors, tracebacks, warnings)          |
| `output_log_4732.txt` | **stdout** (printed info, progress bars, training status) |

📂 These are located in:

```
/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/
```

You can confirm:

```bash
ls -lh error_log_4732.txt output_log_4732.txt
```

To **watch logs live**:

```bash
tail -f output_log_4732.txt
tail -f error_log_4732.txt
```

---

### ✅ 2. **Checkpoints and training outputs**

Created here:

```
/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/output_<JOBID>/
```

Why? This is controlled by:

```bash
--output_dir /workspace/output
```

…and `/workspace/output` is mapped to:

```bash
-v /projects/data/.../output_${SLURM_JOBID}:/workspace/output
```

So if your job ID was `4732`, you’ll find checkpoints in:

```
output_4732/
```

📂 Inside that folder, typical files include:

| Filename / Folder                     | Meaning                      |
| ------------------------------------- | ---------------------------- |
| `checkpoint-200/`                     | Saved model after 200 steps  |
| `checkpoint-400/`                     | Saved model after 400 steps  |
| `trainer_state.json`                  | HuggingFace trainer metadata |
| `pytorch_model.bin` (or shards)       | Main model weights           |
| `config.json`, `tokenizer.json`, etc. | Tokenizer and model config   |

---

### 🔍 **Example:**

```bash
cd /projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/output_4732/
ls
```

Might show:

```
checkpoint-200/
checkpoint-400/
pytorch_model.bin
trainer_state.json
```

Or, inside a specific checkpoint:

```bash
ls checkpoint-400/
```

Output:

```
config.json
pytorch_model.bin
trainer_state.json
tokenizer.json
```

---

## 🎯 How to know **which files to check?**

| Goal                             | File to check                       | Command                              |
| -------------------------------- | ----------------------------------- | ------------------------------------ |
| Check training progress          | `output_log_<jobid>.txt`            | `tail -f output_log_4732.txt`        |
| Check for errors                 | `error_log_<jobid>.txt`             | `grep -i error error_log_4732.txt`   |
| Confirm checkpoint was saved     | `output_<jobid>/checkpoint-*/`      | `ls output_4732/`                    |
| Check model config / HF metadata | `output_<jobid>/trainer_state.json` | `cat output_4732/trainer_state.json` |

---

## 🔧 Extra Tip: Check logs inside job

If your job is still running:

```bash
squeue -u $USER
```

And to open a terminal to your node (if allowed):

```bash
srun --jobid=<jobid> --pty bash
```

You can then:

```bash
cd /workspace/output
ls -lh
```

To see checkpoints *inside* the container (same as host `output_4732/`).

---

## Summary Table

| What                    | Path example                                         | Description                         |
| ----------------------- | ---------------------------------------------------- | ----------------------------------- |
| SLURM logs (stdout)     | `output_log_4732.txt`                                | Print statements, progress, metrics |
| SLURM logs (stderr)     | `error_log_4732.txt`                                 | Errors, warnings, tracebacks        |
| Training checkpoints    | `output_4732/checkpoint-200/`                        | Model state after 200 steps         |
| Final model             | `output_4732/pytorch_model.bin` or latest checkpoint | Final output                        |
| Metadata/config/trainer | `output_4732/trainer_state.json`                     | Tracks epoch, steps, etc.           |
| Training folder (host)  | `/projects/data/.../output_4732/`                    | All outputs land here               |

--- 

---

## 1. High-level flow (what happens, in order)

1. **SLURM reserves resources** (GPUs, RAM, CPUs, nodes).
2. **Environment variables** are exported so every node agrees on where to cache models, how to talk to each other, and how many total GPU processes will run.
3. **Docker image is loaded** (`swift.tar`) on every node and any dangling training container from a previous run is force-removed.
4. **A fresh output folder** named `output_<job-id>` is created on the head node.
5. **A new container starts** on every node (`--net=host`, `--gpus all`).

   * Inside it a venv is activated, then
   * `torchrun` launches 8 GPU workers per node, wired together with the right **MASTER\_ADDR/PORT** so PyTorch DDP and DeepSpeed Zero-2 can talk.
6. **`swift/cli/sft.py` fine-tunes Molmo-7B-O-0924** on your formatted synthetic-QA dataset.
7. **Checkpoints, logs, and W\&B metrics** are streamed out to mounted host folders.
8. When training ends (or the job is cancelled) the container auto-removes (`--rm`).

---

## 2. Directory & volume map (host → container)

| Host path                                                              | Container path                          | Purpose                                       |
| ---------------------------------------------------------------------- | --------------------------------------- | --------------------------------------------- |
| `/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift` | `/workspace`                            | Code repo root                                |
| (same) `/output_<JOBID>`                                               | `/workspace/output`                     | **All checkpoints & evaluation artifacts**    |
| `/projects/data/vision-team`                                           | `/projects/data/vision-team`            | Large shared datasets (read-only in practice) |
| Docker image file `swift.tar` (same folder as script)                  | *loaded into* `swift_img:tr_and_wb_fix` | Training image                                |

> **Tip :** Everything under `/workspace/output` inside the container is immediately visible on the host in `.../output_<JOBID>`—handy for rsyncing or tail-f-ing checkpoints.

---

## 3. Where files end up

| File / pattern                         | Location on the **login/head node**                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------- |
| SLURM stdout/stderr                    | `error_log_<jobid>.txt`, `output_log_<jobid>.txt` in the directory where you ran `sbatch`   |
| Training checkpoints (every 200 steps) | `GR_Model_Training_with_Swift/output_<jobid>/checkpoint-<step>/`                            |
| Final HF model                         | Same folder, last checkpoint or `pytorch_model.bin` if `save_strategy` auto-consolidates    |
| W\&B run                               | Appears on [https://wandb.ai](https://wandb.ai) under your account keyed by `WANDB_API_KEY` |

---

## 4. Line-by-line explanation

### 4.1 SLURM header

```bash
#SBATCH --partition=defq           # Which queue/partition to submit to
#SBATCH --nodelist=bharatgpt158    # Explicit host; remove for auto-allocation
#SBATCH --job-name=grvqa_swift...  # Shows up in squeue
#SBATCH --mem-per-cpu=4G           # 4 GB RAM per vCPU
#SBATCH --cpus-per-task=64         # Each node gets 64 logical CPUs
#SBATCH --gres=gpu:8               # 8 GPUs per node
#SBATCH --exclusive                # Block others from sharing the node
#SBATCH --error=error_log_%j.txt   # %j = JOBID
#SBATCH --output=output_log_%j.txt
#SBATCH --mail-user=... --mail-type=ALL
```

*If you want more nodes, replace `--nodelist=` with `--nodes=N` (plus optional `--ntasks-per-node=1`).*

### 4.2 User-tunable variables

```bash
export USERNAME="venkat_kesav"          # Used in paths below
export MODELSCOPE_CACHE="/workspace/models"  # Inside container
```

### 4.3 Figure out the cluster topology

```bash
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
head_node=${nodes_array[0]}                     # First host = master
head_node_ip=$(srun ... hostname -I | grep '^10\.' | head -n1)
head_node_port=$((10000 + SLURM_JOBID % 10000)) # Avoid port collisions
```

### 4.4 Distribute the Docker image

```bash
srun ... docker load -i ./swift.tar            # Every node loads image locally
```

### 4.5 Set distributed-training env vars

```bash
export WORLD_SIZE=$(($SLURM_NNODES * 8))       # 8 GPUs * nodes
export MASTER_ADDR=${head_node_ip}
export MASTER_PORT=${head_node_port}
export NCCL_DEBUG=INFO ...                     # Verbose NCCL/Gloo
```

### 4.6 Pre-run housekeeping

```bash
srun ... mkdir output_${SLURM_JOBID}           # Checkpoint sink
srun ... docker rm -f kesav_ddp_train_swift || true
```

### 4.7 Launch the training job

```bash
docker run --shm-size=32g --rm --net=host --gpus all \
  --name kesav_ddp_train_swift \
  -e WORLD_SIZE ...                            # pass-through envs
  -v hostdir:/workspace ...                    # two main mounts
  swift_img:tr_and_wb_fix                      # <-- IMAGE NAME
```

Inside the container:

```bash
source /home/venkat_kesav/.venv/bin/activate   # Activate Python venv
torchrun --master_port ${MASTER_PORT} \
         --nproc_per_node=8 --nnodes=${SLURM_NNODES} \
         --master_addr=${MASTER_ADDR} \
         --node_rank=${SLURM_NODEID} \
         /home/venkat_kesav/ms-swift/swift/cli/sft.py ...  # actual Swift CLI
```

Key CLI flags:

| Flag                                       | Meaning                       |
| ------------------------------------------ | ----------------------------- |
| `--model allenai/Molmo-7B-O-0924`          | Base checkpoint from HF hub   |
| `--train_type full`                        | Full-parameter SFT (no LoRA)  |
| `--dataset .../swift_training_format_3#10` | Local dataset folder          |
| `--torch_dtype bfloat16`                   | Use bf16 to save VRAM         |
| `--save_strategy steps --save_steps 200`   | Snapshot every 200 iterations |
| `--report_to wandb`                        | Real-time dashboards          |

---

## 5. How to **scale to more nodes**

1. **SLURM line**: change

   ```bash
   #SBATCH --nodelist=bharatgpt158
   ```

   to something like

   ```bash
   #SBATCH --nodelist=bharatgpt158,bharatgpt159,bharatgpt160
   ```
2. **Nothing else**: the script uses SLURM’s environment (`$SLURM_NNODES`, `$SLURM_NODEID`) to set `WORLD_SIZE` and `node_rank` automatically.

---

## 6. Customising paths & image names

| Want to change…      | Edit this…                                                                                                                                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Docker image tag** | `swift_img:tr_and_wb_fix` (in the `docker run` line)                                                                                                                                                        |
| **Container name**   | `--name kesav_ddp_train_swift`                                                                                                                                                                              |
| **Model cache dir**  | `export MODELSCOPE_CACHE=...` (must exist inside container)                                                                                                                                                 |
| **Dataset location** | `--dataset /path/to/your/data`                                                                                                                                                                              |
| **Output dir root**  | the mount `-v /projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift:/workspace` – leave the right-hand side `/workspace` but point the left-hand side to any scratch space with enough quota |

---

## 7. Troubleshooting quick hits

| Symptom                                | Likely cause                        | Fix                                                                               |
| -------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------- |
| Containers never start, job just hangs | port already in use                 | hard-set `head_node_port` to something free                                       |
| `NCCL WARN Failed to connect`          | wrong interface names               | set `NCCL_SOCKET_IFNAME` to `eth0` or your IB device                              |
| `CUDA out of memory` early             | dataset preprocessing spike         | lower `--per_device_train_batch_size` or increase `--gradient_accumulation_steps` |
| Checkpoints missing                    | forgot to create host output folder | script already `mkdir`s; check disk quota                                         |

---

### Sleep-mode summary 😴

> **TL;DR – submit with `sbatch molmo_sft.sh`, watch `squeue`, shrug.**
> SLURM grabs 1 node × 8 GPUs (or more if you ask), loads `swift_img:tr_and_wb_fix`, spins up `kesav_ddp_train_swift`, and `torchrun` fine-tunes Molmo-7B. All logs stream to `error_log_<jobid>.txt` and `output_log_<jobid>.txt`; checkpoints + metrics land in `output_<jobid>/` beside the script. 



