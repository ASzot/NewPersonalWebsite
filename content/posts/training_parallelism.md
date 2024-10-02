---
title: Training Parallelism
date: 2024-08-25
---


## Intro

194,103,296 params

## Background - Simple Training Loop

- Give code for `main.py` but remove all the code under the `is_distrib` guard.
- Give the samples per second and max memory usage.
    - samples_per_second=482,427, max_mem=35,480

## Data Parallel

- Show code modifications to enable DDP. 
- Give 2 GPU speeds
    - samples_per_second=1,060,390, max_mem=35,850
- Give 2 node speeds (but still 2 GPUs). Note that Interconnect is 10GB/s ethernet, not high speed infiniband.
    - TODO

## Sharded Optimizer
- Show the small code modification to start using Zero optimizer.
- Show the reduction in memory at not reduction in speed. Talk about how it's possible.

## FSDP
- With the smaller model setting $E=1024,H=8,L=16$ we have 194,103,296 parameters. This results in 34.65gb mem usage.
- The formula for our memory usage is:
```python
toks = 50257 # for gpt2 tokenizer.
bytes_per_param = 2 # fixed for bfloat16
model_params = ... # just get this from the model in pytorch, no need to manually calculate it.
activations = B*C*(2*toks+l*(6*e+h*C))
training_bytes = (4 * model_params + 2 * activations) * bytes_per_param / 1_000_000_000
# Quick check in python
f"{(4 * model_params + 2 * B*C*(2*toks+l*(6*e+h*C))) * bytes_per_param / 1_000_000_000:,}"
# Only model stats
f"{(4 * model_params) * bytes_per_param / 1_000_000_000:,}"
```
Where the `6*e` comes from:
- 3 from QKV projections
- 2 from FFN
- 1 from linear projection

Where the `4*model_params` comes from
- Model params
- Gradients
- Momentum
- Variance

Where the `2*acts` comes from forward and backwards.

- `python main.py --embed-dim 2048 --n-heads 32 --n-layers 32` results in 4,769,521,664 params. 37GB for model parameters. 
- `--embed-dim 1024 --n-heads 32 --n-layers 32 --fsdp` is 336_742_400 params. 

## Acknowledgements
- https://github.com/pytorch/torchtune useful for understanding how FSDP is applied.
    - https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py in particular
- https://pytorch.org/docs/stable/fsdp.html FSDP setting explanations.
