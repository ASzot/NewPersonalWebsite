---
title: Training Parallelism
date: 2024-08-25
---

## Intro

This post shows how to scale training transformer models in terms of batch size, model size and sequence length. First, this post goes over scaling the batch size with multi-GPU and multi-node training. Next, the post details how to scale the model size with training a 7B parameter LLM where training does not fit on a single GPU, but can be trained on 2 GPUs. Finally, the post discusses scaling to a 10,000 token context length when the activations do not on a single GPU.

The code is concise and implemented in pure PyTorch with the potential for scaling to multi-GPU, multi-node, big parameter counts, and long sequence lengths. The code for this post is at [github.com/ASzot/training-parallelism](https://github.com/ASzot/training-parallelism). 

## Background - Simple Training Loop
We will use the Llama-2 transformer architecture which uses RoPE positional embeddings, RMSNorm, and SwiGLU activations in the feed forward network with a 1.5x larger hidden dimension. The model code is at [this link](https://github.com/ASzot/training-parallelism/common.py) and is omitted from this post. In addition, to creating a model with this architecture, the training script also creates a tokenizer using the `tiktok` library and sets up a simple dataset iterator for the TinyStories dataset. The TinyStories dataset is only a 1.2GB text file and is automatically downloaded if not present. We only run 10 updates just to analyze the training speeds and GPU usage.
```python
tokenizer = tiktoken.get_encoding("gpt2")
device = "cuda"
model = (
    LanguageModel(
        vocab_size=tokenizer.n_vocab,
        max_ctx_len=cfg.context_len,
        embed_dim=cfg.embed_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        tensor_parallel_size=cfg.tensor_parallel_size,
    )
    .to(torch.bfloat16)
    .to(device)
)
print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Load the Tiny Stories training dataset. If it is not downloaded, it is
# automatically downloaded.
train_name = "TinyStories-train"
load_dataset(train_name)
# `train_data_gen` is an iterator that returns numpy array of shape
# `[batch_size, context_len]` representing the tokenized data.
train_data_gen = data_generator(
    train_name,
    tokenizer,
    # Sample sequences of length context_len+1 so we have the next token target
    # for prediction.
    seq_length=cfg.context_len + 1,
    batch_size=cfg.batch_size,
)

# Start memory and timing collection statistics.
torch.cuda.reset_peak_memory_stats(device)
start = time.time()

# Run the model training for 10 updates.
TOTAL_UPDATES = 10
for update_i, batch in enumerate(train_data_gen):
    batch = torch.from_numpy(batch).to(device)
    optimizer.zero_grad()

    # Train the model to predict the next token.
    pred = model(batch[:, :-1])
    loss = F.cross_entropy(
        rearrange(pred, "b t d -> (b t) d"), rearrange(batch[:, 1:], "b t -> (b t)")
    )

    loss.backward()
    optimizer.step()

    if rank == 0:
        print(f"Update={update_i}, Loss={loss.item()}")
    if update_i == TOTAL_UPDATES:
        break

# Log the time and amount of GPU memory for the training loop.
tokens_per_second = int(
    (world_size * cfg.batch_size * cfg.context_len) * (time.time() - start)
)
# Memory in GB.
max_mem = torch.cuda.max_memory_allocated(device) / 1_000_000_000
print(f"{tokens_per_second=:,}, {max_mem=:.2f}gb")
```
Then run the following to train with a model with 246M parameters for 10 updates (which only takes a couple seconds to run). For this post, I ran everything on NVIDIA A40 GPUs.
```bash
python main.py embed_dim=1024 n_heads=8 n_layers=16 batch_size=128 context_len=256
```
This runs at 508,829 tokens-per-second with 35.37gb GPU memory use.
 
## Data Parallel - Scaling Batch Size

We need a larger batch size to scale up training, but there is no more room on the GPU to increase the batch size. Data distributed parallel (DDP) solves this issue by training on multiple GPUs. DDP runs the same model under different data batches on each GPU and then averages the gradients to learn from all examples. We use `torchrun` for DDP in PyTorch. `torchrun` launches the same job in multiple processes each running on a separate GPU. `torchrun` sets job information via environment variables with `LOCAL_RANK` for the GPU the process is running on in the current node and `WORLD_SIZE` for the number of workers. To enable DDP, we add the following code to the existing script.
```python
# ... imports (omitted)

from torch.nn.parallel import DistributedDataParallel

# Check if we launched with `torchrun` for distributed training.
is_distrib = dist.is_available() and "LOCAL_RANK" in os.environ

if is_distrib:
    # Fetch information set by torchrun.
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    print(
        f"Starting worker {rank+1}/{world_size} with master address {os.environ['MASTER_ADDR']} on port {os.environ['MASTER_PORT']}"
    )

else:
    # Set info for just a single worker.
    local_rank = 0
    rank = 0
    world_size = 1


if is_distrib:
    dist.init_process_group("nccl")
    # Use the GPU specified by the rank.
    torch.cuda.set_device(local_rank)

# ... model creation code (omitted)

if is_distrib:
    print("Starting data parallel")
    model = DistributedDataParallel(model)

# ... remaining training loop (omitted)
```
Then run the following command to now train on 2 GPUs:
```bash
torchrun --nproc_per_node=2 main.py embed_dim=1024 n_heads=8 n_layers=16 batch_size=128 context_len=256
```
And this gives 1,055,883, tokens-per-second with 35.86gb GPU memory use. In addition to scaling to multiple GPUs on a single node, `torchrun` and DDP allow scaling to multiple nodes without any code changes. Say you have two nodes with IP addresses `node1.com` and `node2.com`. Then on both of the nodes run:
```bash
torchrun --nproc_per_node=1 --rdzv-id=12345 --rdzv-backend=c10d --rdzv-endpoint=node1.com:29401 --nnodes=2 main.py
```
This starts training with 1 GPU on both of the nodes. The choice of `node1` for the endpoint was arbitrary. Port 29401 was also arbitrary, but needs to be open on both machines. The ID of 12345 can be anything, but needs to be shared between all of the jobs. Typically jobs are launched via a cluster scheduling program like Slurm. Slurm provides info about endpoints and job IDs that can be used to set this information. 

## Sharded Optimizer - Save GPU Memory

In standard DDP, everything is replicated between each GPU, meaning the gradients and optimizer state are also replicated across GPUs. However, this is redundant, the gradients and optimizer states can be partitioned per-device with no additional communication overhead. How this generally works is that the each device computes gradients or optimizer states for the entire model for its current data batch. The state is then set to the partition that owns that partition of the gradients or optimizer states. The data from all devices is then reduced on the GPU that owns the partition. This technique is described in more detail [ZeRO paper](https://arxiv.org/abs/1910.02054) and to use it with PyTorch, simply change the optimizer setup to:
```python
# ... Rest of code omitted
from torch.distributed.optim import ZeroRedundancyOptimizer

if cfg.zero_opt:
    optimizer = ZeroRedundancyOptimizer(
        model.parameters(), optimizer_class=torch.optim.Adam, lr=3e-4
    )
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# ... Rest of code omitted
```
Now running training on 2 GPUs as before with:
```bash
torchrun --nproc_per_node=2 main.py embed_dim=1024 n_heads=8 n_layers=16 batch_size=128 context_len=256 zero_opt=True
```
This gives 35.37gb GPU memory use, which is a slight memory saving over the previous result. The memory savings for this technique are greater as the model size, and thus the optimizer and gradients increase in size.

## Fully Sharded Data Parallel - Scaling Model Size

Now we will scale training to a 7.2 billion parameter model. This requires setting the parameters `embed_dim=4096 n_heads=32 n_layers=48` for launching the job. Is this model small enough to fit on the A40 GPU which has 46gb of GPU memory? To find out, first calculate that the memory for the model and the gradients will be `4 * num_model_params` (parameters, gradients, Adam momentum, Adam variance). The memory for the activations of the model is is `6 * embed_dim + heads*context_len` for each transformer layer where the 6 comes from the 3 QKV projections, 2 from the FFN, and 1 from the final linear projection. This is only a rough estimate as it ignores the expansion factor in the Llama FFN layer. There is also an added `2 * num_tokens` for the embedding and output layers. This is then scaled by the number of tokens in the batch which is `batch_size * context_len`. We thus have:
```python
num_tokens = 50257 # The number of tokens for the GPT-2 tokenizer we are using.
num_model_params = 7_257_206_784 # Retrieved from creating the model.
activations = batch_size * context_len * (2*num_tokens+n_layers*(6*embed_dim+n_heads*context_len))
# Multiply by 2 for 2 bytes per-param for bfloat16 format.
training_mem_gb = (4 * num_model_params + 2 * activations) * 2 / 1_000_000_000
print(f"{training_mem_gb:,}")
```
This gives 59.77gb, greater than the 46gb of A40 GPU memory, so no, the training will not fit on the GPU. To resolve this, we partition the model parameters, meaning the parameters of the model are split up between the GPUs. We can then train larger models by using more GPUs. Fully sharded data parallel (FSDP) in PyTorch provides this functionality. This is implemented via the following code:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

if cfg.fsdp:
    print("Starting FSDP")
    model = FSDP(
        model,
        auto_wrap_policy=ModuleWrapPolicy({CausalTransformerLayer}),
    )
elif is_distrib:
    print("Starting data parallel")
    model = DistributedDataParallel(model)
```
`auto_wrap_policy=ModuleWrapPolicy({CausalTransformerLayer})` indicates what is sharded between the GPUs and is necessary to specify in this case. Printing the resulting model we see:
```
FullyShardedDataParallel(
  (_fsdp_wrapped_module): LanguageModel(
    (tok_embed): Embedding(50257, 4096)
    (rope): RoPE_Embedding()
    (blocks): Sequential(
      (0): FullyShardedDataParallel(
        (_fsdp_wrapped_module): CausalTransformerLayer(
          (qkv_proj): Linear(in_features=4096, out_features=12288, bias=False)
          (att_out_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (att_norm): RMSNorm()
          (ffn_norm): RMSNorm()
          (ffn): FFN(
            (up_proj): Linear(in_features=4096, out_features=6144, bias=False)
            (gate_proj): Linear(in_features=4096, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=4096, bias=False)
          )
          (rope): RoPE_Embedding()
        )
      )
      (1): FullyShardedDataParallel(
        (_fsdp_wrapped_module): CausalTransformerLayer(
          (qkv_proj): Linear(in_features=4096, out_features=12288, bias=False)
          (att_out_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (att_norm): RMSNorm()
          (ffn_norm): RMSNorm()
          (ffn): FFN(
            (up_proj): Linear(in_features=4096, out_features=6144, bias=False)
            (gate_proj): Linear(in_features=4096, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=4096, bias=False)
          )
          (rope): RoPE_Embedding()
        )
      )
      ...
```
`FullyShardedDataParallel` wrapping each individual layer means that these layers are what is sharded between GPUs. If `auto_wrap_policy` is not specified, only the external `LanguageModel` would be wrapped in `FullyShardedDataParallel`, meaning the individual layers are sharded between GPUs and thus none of the model parameters are sharded, resulting in no memory savings. Another important aspect of FSDP is that everything must be a `nn.Module` and all model forward logic must take place in the `forward` method, rather than a function that operates on the module, because FSDP will wrap the module in the `FullyShardedDataParallel` wrapper. To see the benefits of FSDP, run the following:
```python
torchrun --nproc_per_node=2 main.py fsdp=True embed_dim=4096 n_heads=32 n_layers=48 batch_size=1
```
Which gives 13,111 tokens per second and 36.33gb GPU memory usage for training the 7b parameter model, whereas before the model training ran out of memory. By turning off the model partitioning, FSDP also turns into regular DDP. This is achieved by setting:
```python
sharding_strategy = (
    ShardingStrategy.FULL_SHARD
    if cfg.shard_strategy == "full"
    else ShardingStrategy.NO_SHARD
)
model = FSDP(
    model,
    sharding_strategy=sharding_strategy,
    auto_wrap_policy=ModuleWrapPolicy({CausalTransformerLayer}),
)
```


## Sequence and Tensor Parallel - Scaling Context Length

The previous models were trained with a short context length of 256. How can we scale training to a longer context length of 10,000 without running out of memory? Running the following results in OOM on 2xA40 GPUs:
```bash
torchrun --nproc_per_node=2 main.py context_len=10000 embed_dim=1024 batch_size=4 fsdp=True shard_strategy=no
```
The solution is to use _sequence and tensor parallelism_ where matrix multiplications are partitioned per GPU. Section 3 of the [Megatron-LM paper](https://arxiv.org/abs/1909.08053) describes the idea best, and I will briefly summarize the idea here. Let $ N$ be the sequence length, $D $ the hidden dimension, $ X \in \mathbb{R}^{N \times D}$ the input tensor (ignoring batch dimension) and $ W \in \mathbb{R}^{D \times D}$ be weight parameters for a linear layer. Then the linear projection $ X W $ can be rewritten by first splitting $ W$ "columnwise" into $ W = \left[ W_1, W_2 \right] $ where $ W_1 \in \mathbb{R}^{D \times D_1}, W_2 \in \mathbb{R}^{D \times D_2}$ and then computing $ \left[ X W_1, X W_2 \right] $ where  these two operations can be done in parallel per device and the result is the same as $ XW$. 

To apply this idea to the attention operation, we columnwise partition the query, key, and value projections $ Q, K, V \in \mathbb{R}^{D \times D} $ into $ Q = [Q_1, Q_2], K = [K_1, K_2], V = [V_1, V_2] $. For input $ X \in \mathbb{R}^{N \times D}$, we then compute attention as normal (ignoring multi-head outputs in the notation) with $ A_i = \text{softmax}\left( \frac{(XQ_i) (XK_i)^\top}{\sqrt{D_i}} \right) V_i \in \mathbb{R}^{N \times D_i}$. Each $ A_i $ only involves hidden dimensions $ D_i$ and can be computed independently of other $ A_i$. For the output projection, $ W^{O} \in \mathbb{R}^{D \times D}$ we could reduce $ \left[   A_1, A_2 \right]  \rightarrow  A$ and then compute $ A W^{O}$, but we can avoid this reduce by computing $ A_i W^{O}_i$ separately where $ W^{O}_i \in \mathbb{R}^{D_i \times D}$  is now partitioned **"rowwise"**. Finally, we reduce outputs $ \left[ A_1 W^{O}_1, A_2 W^{O}_2 \right] $. This description was for 2 partition elements, but the same technique works exactly the same for more partition elements. Tensor parallelism is also applied in the same way to the FFN network. 

In PyTorch, tensor parallelism works by setting up a device mesh, which is like the previous `dist.init_process_group` call, but with more control over the setup. The distributed setup code is now:
```python
if cfg.fsdp and use_tp:
    # This must be before the model creation to properly set the GPU device ID
    dp_size = world_size // cfg.tensor_parallel_size
    tp_size = cfg.tensor_parallel_size
    # This is in place of `init_process_group` for more complex sharding
    # strategies.
    device_mesh = init_device_mesh(
        device,
        (dp_size, tp_size),
        # The mesh names are arbitrary. The only things that matters is how we
        # use these meshes. The "dp" mesh is passed to FSDP and
        # the "tp" mesh is passed to the model parallelization.
        mesh_dim_names=("dp", "tp"),
    )
elif is_distrib:
    dist.init_process_group("nccl")
    # Use the GPU specified by the rank.
    torch.cuda.set_device(local_rank)
    device_mesh = None
elif use_tp:
    raise ValueError(f"fsdp must be `True` when setting {cfg.tensor_parallel_size=}.")
``` 
After initializing the model, we parallelize the attention and FFN layers of each attention block in the model. Furthermore, the RMSNorm normalization operation can also be sharded over the sequence dimension. The following code sets up this parallelization.
```python
tp_mesh = device_mesh["tp"]
fsdp_device_mesh = device_mesh["dp"]

# Keys of the TP plan must be the same module names in `model` (as how they
# are saved in the `state_dict` for example).
model = parallelize_module(
    module=model,
    device_mesh=tp_mesh,
    parallelize_plan={
        # Partition the input embeddings (each row is a different token).
        "tok_embed": RowwiseParallel(
            # For each module, we specify how the inputs will be sharded
            # where `Replicate()` means not sharded (they are repeated on
            # every device), and `Shard(1)` means they are sharded along
            # sequence dimension.
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        "output_norm": SequenceParallel(),
        "lm_head": ColwiseParallel(
            input_layouts=Shard(1), output_layouts=Replicate()
        ),
    },
)

# Parallelize each transformer layer.
for block in model.blocks:
    parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan={
            "att_norm": SequenceParallel(),
            "ffn_norm": SequenceParallel(),
            # Attention.
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "attention.q_proj": ColwiseParallel(),
            "attention.k_proj": ColwiseParallel(),
            "attention.v_proj": ColwiseParallel(),
            "attention.att_out_proj": RowwiseParallel(output_layouts=Shard(1)),
            # FFN.
            "ffn": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "ffn.up_proj": ColwiseParallel(),
            "ffn.gate_proj": ColwiseParallel(),
            "ffn.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        },
    )
```
To work with these different sharding strategies between the different modules, we specify the sharding format the the input and output of the Modules via `input_layouts` and `desired_input_layouts`. `Replicate()` means the tensor is not sharded and replicated per-device. `Shard(dim=1)` means the tensor is sharded, and on the sequence dimension, which is what the `dim=1` refers to (recall the tensors have shape $B \times N \times D$). The FSDP initialization also must be slightly modified to account for the new data parallel device mesh:
```python
if cfg.fsdp:
    print("Wrapping model in fsdp")
    sharding_strategy = (
        ShardingStrategy.FULL_SHARD
        if cfg.shard_strategy == "full"
        else ShardingStrategy.NO_SHARD
    )
    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=ModuleWrapPolicy({CausalTransformerLayer}),
        ####################
        # For TP
        device_mesh=fsdp_device_mesh,
        use_orig_params=False,
        ####################
    )
elif is_distrib:
    print("Starting data parallel")
    model = DistributedDataParallel(model)
```
Now we are able to train with a sequence length of 10,000 by using a tensor parallelism factor of 2 without running out of GPU memory.
```bash
torchrun --nproc_per_node=2 main.py context_len=10000 embed_dim=1024 batch_size=4 fsdp=True shard_strategy=no tensor_parallel_size=2
```
This gives 1,992,486 tokens per second and 30.85gb GPU usage. 

## Conclusion

The following techniques are combined into a compact [codebase here](https://github.com/ASzot/training-parallelism) that scales to large batch sizes, model parameter counts, and sequence lengths across multiple GPUs and nodes.

## Acknowledgements
- https://pytorch.org/docs/stable/fsdp.html FSDP setting explanations.
- https://pytorch.org/tutorials/intermediate/TP_tutorial.html Useful example of applying tensor parallelism to a language model.

