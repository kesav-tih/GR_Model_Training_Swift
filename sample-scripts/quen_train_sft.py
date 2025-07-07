# train_qwen_lora.py

import os
from functools import partial
from swift.llm import (
    get_model_tokenizer, load_dataset, get_template, EncodePreprocessor
)
from swift.utils import (
    get_logger, find_all_linears, get_model_parameter_info,
    plot_images, seed_everything
)
from swift.tuners import Swift, LoraConfig
from swift.trainers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
# from IPython.display import display

# ========== Setup ==========
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = get_logger()
seed_everything(42)

# ========== Configuration ==========
# Model settings
model_id_or_path = '/workspace/models/Qwen2.5-VL-7B-Instruct/'
system = 'You are a helpful assistant.'
output_dir = os.path.abspath(os.path.expanduser('output'))

# Dataset settings
dataset = [
    'swift/self-cognition#500'
]
data_seed = 42
max_length = 2048
split_dataset_ratio = 0.01
num_proc = 4
model_name = ['小黄', 'Xiao Huang']
model_author = ['魔搭', 'ModelScope']

# LoRA config
lora_rank = 8
lora_alpha = 32

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_checkpointing=True,
    weight_decay=0.1,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    report_to=['wandb'],
    logging_first_step=True,
    save_strategy='steps',
    save_steps=50,
    eval_strategy='steps',
    eval_steps=50,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    metric_for_best_model='loss',
    save_total_limit=2,
    logging_steps=5,
    dataloader_num_workers=1,
    data_seed=data_seed,
)

logger.info(f'Output directory: {output_dir}')

# ========== Load Model ==========
model, tokenizer = get_model_tokenizer(model_id_or_path)
logger.info(f'Model info: {model.model_info}')

template = get_template(
    model.model_meta.template,
    tokenizer,
    default_system=system,
    max_length=max_length
)
template.set_mode('train')

# ========== LoRA Configuration ==========
target_modules = find_all_linears(model)
lora_config = LoraConfig(
    task_type='CAUSAL_LM',
    r=lora_rank,
    lora_alpha=lora_alpha,
    target_modules=target_modules
)
model = Swift.prepare_model(model, lora_config)
logger.info(f'LoRA config: {lora_config}')
logger.info(f'Model with LoRA applied: {model}')

# Log trainable parameters
model_parameter_info = get_model_parameter_info(model)
logger.info(f'Model parameter info: {model_parameter_info}')

# ========== Load and Preprocess Dataset ==========
train_dataset, val_dataset = load_dataset(
    dataset,
    split_dataset_ratio=split_dataset_ratio,
    num_proc=num_proc,
    model_name=model_name,
    model_author=model_author,
    seed=data_seed
)

logger.info(f'Train dataset sample: {train_dataset[0]}')

train_dataset = EncodePreprocessor(template=template)(train_dataset, num_proc=num_proc)
val_dataset = EncodePreprocessor(template=template)(val_dataset, num_proc=num_proc)

logger.info(f'Encoded train dataset sample: {train_dataset[0]}')

# Print tokenized example
template.print_inputs(train_dataset[0])

# ========== Trainer ==========
model.enable_input_require_grads()
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=template.data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    template=template,
)

# Start training
trainer.train()

# Save checkpoint info
last_model_checkpoint = trainer.state.last_model_checkpoint
logger.info(f'Last model checkpoint: {last_model_checkpoint}')

# # ========== Visualize Loss ==========
# images_dir = os.path.join(output_dir, 'images')
# logger.info(f'Loss images directory: {images_dir}')

# plot_images(
#     images_dir,
#     training_args.logging_dir,
#     metrics=['train/loss'],
#     smoothing_factor=0.9
# )

# # Display loss image
# image_path = os.path.join(images_dir, 'train_loss.png')
# if os.path.exists(image_path):
#     image = Image.open(image_path)
#     display(image)
# else:
#     logger.warning(f"Loss image not found at: {image_path}")
