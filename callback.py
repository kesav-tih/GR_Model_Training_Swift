# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments, IntervalStrategy
import os 
import torch 

from swift.utils import get_logger

logger = get_logger()


class EarlyStopCallback(TrainerCallback):
    """An early stop implementation"""

    def __init__(self, total_interval=3):
        self.best_metric = None
        self.interval = 0
        self.total_interval = total_interval

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        operator = np.greater if args.greater_is_better else np.less
        if self.best_metric is None or operator(state.best_metric, self.best_metric):
            self.best_metric = state.best_metric
        else:
            self.interval += 1

        if self.interval >= self.total_interval:
            logger.info(f'Training stop because of eval metric is stable at step {state.global_step}')
            control.should_training_stop = True

def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


class EpochExportCallback(TrainerCallback):
    """Write EPOCH to env + file once per epoch (rank‑0 only)."""

    def on_epoch_begin(self, args, state, control, **kwargs):
        if get_rank() != 0:
            return  # other ranks skip
        epoch_id = int(state.epoch)
        os.environ["EPOCH"] = str(epoch_id)  # visible to child procs
        print(f"[Rank‑0] Exported EPOCH={epoch_id}")

class PklIdEnvCallback(TrainerCallback):
    """
    Sets $PKL_ID to either 'epoch{N}' or 'step{N}' right before evaluation.
    """

    def on_evaluate(self, args, state, control, **kwargs):
        print("on evaluate")
        print("args getting printed here", args)
        print("state", state)
        if args.eval_strategy == IntervalStrategy.STEPS:
            os.environ["PKL_ID"] = f"step{state.global_step}"
        else:  # "epoch"
            # `state.epoch` is a float like 1.0, cast to int
            os.environ["PKL_ID"] = f"epoch{int(state.epoch)}"

extra_callbacks = [PklIdEnvCallback()]
#extra_callbacks = [EpochExportCallback()]
# This example shows a simple example of EarlyStop Callback, uncomment this to use
# extra_callbacks = [EarlyStopCallback()]
