from transformers.utils import logging

logger = logging.get_logger("transformers")


def model_to_half(model, cast_model=True):
    if cast_model:
        model.half()

    for n, m in model.named_modules():
        if m.__class__.__name__ == 'QuantLinear':
            logger.debug(f'Converting to half {n}.')
            m.scales = m.scales.half()
            m.bias = m.bias.half() if (m.bias is not None) else None
    logger.info('Converted as Half.')


import torch
import time
from transformers import TrainerCallback
import numpy as np

MB = 1024 * 1024

def compute_tensor_size(x):
    if x.dtype in [torch.float32, torch.int]:
        ret = np.prod(x.size()) * 4 
    elif x.dtype in [torch.bfloat16, torch.float16, torch.int16]:
        ret = np.prod(x.size()) * 2
    elif x.dtype in [torch.int8]:
        ret = np.prod(x.size()) * 1
    else:
        print(f"[Error] Unsupport data type {x.dtype}")
    return ret

class LogMemoryCallback(TrainerCallback):
    def __init__(self, outfile="out"):
        super().__init__()
        self.peak_mems = []
        self.times = []
        self.iter = 0
        self.outfile = outfile
        with open(self.outfile, "a") as f:
            f.write(
                "model, micro_batch_size, global_batch_size, peak_mem, iter_latency\n")

    def on_step_begin(self, args, state, control, model, tokenizer, optimizer, **kargs):
        torch.cuda.synchronize()
        self.start = time.time()

    def on_step_end(self, args, state, control, model, tokenizer, optimizer, **kargs):
        torch.cuda.synchronize()
        elapsed = time.time() - self.start
        self.times.append(elapsed)
        self.iter += 1
        if self.iter == 2:
            # export data
            with open(self.outfile, "a") as f:
                global_batch_size = args.per_device_train_batch_size * \
                    args.gradient_accumulation_steps
                f.write(
                    f"{model.main_input_name}, {args.per_device_train_batch_size}, {global_batch_size}, {np.median(self.peak_mems)}, {np.median(self.times)}\n")
            exit(0)

    def on_substep_end(self, args, state, control, model, tokenizer, optimizer, **kargs):
        peak_mem = torch.cuda.max_memory_allocated()
        self.peak_mems.append(peak_mem)
        print(f"Peak memory: {peak_mem / MB} MB")

        param_size = 0
        grad_size = 0
        exp_avg_size = 0  # Exponential moving average of gradient values
        exp_avg_sq_size = 0  # Exponential moving average of squared gradient values
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                param_size += compute_tensor_size(p.data)
                grad_size += compute_tensor_size(p.grad.data)
                state = optimizer.state
                if "exp_avg" in state[p]:
                    exp_avg_size += compute_tensor_size(state[p]["exp_avg"])
                if "exp_avg_sq" in state[p]:
                    exp_avg_sq_size += compute_tensor_size(
                        state[p]["exp_avg_sq"])

        print(f"Gradient memory: {grad_size / MB} MB")
        print(f"Exp Avg memory: {exp_avg_size / MB} MB")
        print(f"Exp Avg Sq memory: {exp_avg_sq_size / MB} MB")
