---
title: KV Cache
date: 2024-06-25
---


## Intro
The Key-Value (KV) cache is used in transformer inference to generate the next token faster. For example, when large language models (LLMs) are generating text, the KV cache stores information about the model's hidden activations for the previously generated text to efficiently generate the next text token. While not used during training, the KV cache is a crucial implementation detail for fast transformer inference. In this post, I show that a KV cache results in a $20\times$ inference speedup over naive transformer inference. 

This post goes over a self-contained and minimal KV cache implementation in only PyTorch. I illustrate the ideas with a small language model I trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The code is at [github.com/aszot/kv-cache](https://github.com/ASzot/kv-cache) and is only a single small Python file. It also runs all the below examples in just a couple seconds on a consumer grade GPU.

## KV Cache Formulation
This section formalizes the KV Cache implementation. First consider an input sequence of $n$ input tokens $\{ c_1, \dots, c_n \}$ (like a prompt) and we want to generate a response of $k$ tokens (like an answer to the prompt). These $n$ tokens are embedded into $d$ dimensional vectors $ X_{1:n} \in \mathbb{R}^{n \times d}$. The transformer layer is parameterized by:

- Query, key and value projections $W^Q, W^K, W^V$ all in $\mathbb{R}^{d \times d}$ where for simplicity we assume the key and value hidden dimension are the same as the embedding dimension and there is only one attention head.
- Attention output projection weight $W^O \in \mathbb{R}^{d \times d}$.
- Feedforward network (FFN), which is typically a 2-layer MLP. 

We compute the transformer layer output $T(X_{1:n})$ as:
$$
\text{Att}(X_{1:n}) = \text{Softmax} \left( \frac{Q_{1:n}K_{1:n}^\top}{\sqrt{d}} \right) V_{1:n}
$$
$$
T(X_{1:n}) = \text{FFN} \left( \text{LayerNorm} \left( X_{1:n} + \text{Att}(\text{LayerNorm}(X_{1:n})) W^O \right)  \right)
$$
Where $Q_{1:n} = X_{1:n}W^Q, K_{1:n} = X_{1:n}W^K, V_{1:n} = X_{1:n}W^V$. Notice that $\text{Att}(X_{1:n}) $ is shape $n \times n$ and $Q_{1:n}, K_{1:n}, V_{1:n}$ are all shape $ n \times d$. We then iteratively apply $L$ more transformer layers to get the final activations $X_{1:n}^L$. We predict the next token $ c_{n+1}$ based on $X_n^L$.

The KV cache is used to predict the _next_ token $c_{n+2}$ by reusing the previous computations. The key insight is we only need $X_{n+1}^L$ to predict $c_{n+2}$ (and $X_{1:n}^L $ are anyways the exact same as when computing $c_{n+1}$ due to the causal attention). We embed the previously predicted token $c_{n+1}$ to get $X_{n+1}$ and compute $Q_{n+1}, K_{n+1}, V_{n+1}$ as before. We then compute the attention score of $X_{n+1}$ with $X_{1:n}$ as:
$$
\text{Att}(X_{n+1}) = \text{Softmax} \left( \frac{Q_{n+1} [K_{1:n}, K_{n+1}]^\top}{\sqrt{d}} \right) [V_{1:n}, V_{n+1}]
$$
Where $[A, B]$ indicates concatenating the rows of $A$ and $B$. Notice that $\text{Att}(X_{n+1})$ is shape $1 \times n$. We compute $T(X_{n+1})$ as before and predict $c_{n+2}$ from $X_{n+1}^L$. 

The KV cache for generating token $n+2$ was $K_{1:n}, V_{1:n}$. The benefits of the KV cache are we didn't need to recompute $K_{1:n}, V_{1:n}$ and only need to compute the attention relative to the newly predicted token at $n+1$. Next, we implement this idea in code.

## Model Definition and Forward pass

We will implement the transformer and KV cache primarily with `torch`. `tiktoken` is used for tokenization and `einops` is used to make the attention operation more readable. 

First, define the transformer parameters. The architecture is based on GPT-2. The `TransformerBlock` module defines a single transformer layer and will be stacked to create the final transformer. Each layer initializes the projections $W^Q, W^K, W^V$ (via `qkv_proj`) to produce $Q, K, V$ and the output projection $W^O$ (via `att_out_proj`). The FFN network does not use any bias parameters. We use learned position embeddings. For better efficiency, we also precompute the causal attention mask based on a maximum possible sequence length.
```python
class TransformerBlock(nn.Module):
    """
    Single transformer layer.
    """

    def __init__(self, embed_dim: int, dropout: float, max_ctx_len: int):
        super().__init__()

        # Projections for all of Q, K, V (so multiply by 3)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.att_out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Dropout layers to help generalization (maybe not necessary)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(embed_dim, bias=False)
        self.ln_2 = nn.LayerNorm(embed_dim, bias=False)

        # No bias for the FFN.
        self.ffn = nn.Sequential(
            # Common to use an expanded hidden dimension.
            nn.Linear(embed_dim, 4 * embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=False),
            nn.Dropout(dropout),
        )

        # Causal self-attention mask (at index `i` only attend to `<i`).
        self.mask = torch.tril(torch.ones(max_ctx_len, max_ctx_len)).view(
            1, 1, max_ctx_len, max_ctx_len
        )


class CausalTransformer(nn.Module):
    """
    Full transformer model.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        dropout: float,
        n_layers: int,
        n_heads: int,
        max_ctx_len: int,
    ):
        super().__init__()

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)

        # Learned position embeddings.
        self.pos_embed = nn.Embedding(max_ctx_len, embed_dim)

        # Dropout layer after token and position embeddings.
        self.input_dropout = nn.Dropout(dropout)

        # Track the number of heads for the forward pass.
        self.n_heads = n_heads
        self.embed_dim = embed_dim

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(embed_dim, dropout, max_ctx_len)
                for _ in range(n_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(embed_dim, bias=False)

        # Final layer to predict next token.
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        # Tie the output layer with the input embedding layer (they will have
        # the same weight values).
        self.tok_embed.weight = self.lm_head.weight
```
Next, compute the "parallel" transformer forward pass where we compute the attention scores for all input tokens at the same time. This forward pass is used for training. Note I implement the attention operation to keep the code simple rather than using the built in [PyTorch attention](https://pytorch.org/docs/2.2/generated/torch.nn.functional.scaled_dot_product_attention.html). The next section shows it's inefficient to reuse this code for inference since it recomputes attention scores for every new tokens. The KV cache is the more efficient alternative for inference.

```python
def transformer_forward(model: CausalTransformer, idx):
    ctx_len = idx.shape[1]

    pos_embed = model.pos_embed(torch.arange(0, ctx_len, dtype=int, device=idx.device))

    # Embed input IDs and include position information.
    x = model.tok_embed(idx) + pos_embed
    x = model.input_dropout(x)
    for i, block in enumerate(model.blocks):
        q, k, v = block.qkv_proj(block.ln_1(x)).split(model.embed_dim, dim=2)

        # Convert Q,K,V to shape [batch_size, #heads, context_len, embed_dim]
        q = rearrange(q, "b t (h k) -> b h t k", h=model.n_heads)
        k = rearrange(k, "b t (h k) -> b h t k", h=model.n_heads)
        v = rearrange(v, "b t (h v) -> b h t v", h=model.n_heads)

        att = (q @ k.transpose(-2, -1)) / np.sqrt(k.shape[-1])
        # `att` is shape [batch_size, #heads, context_len, context_len]
        att = att.masked_fill(block.mask[:, :, :ctx_len, :ctx_len] == 0, float("-inf"))

        # Softmax per row.
        att = F.softmax(att, dim=-1)
        att = block.attn_dropout(att)

        y = att @ v
        # Final shape output of attention operator is
        # [batch_size, context_len, (#heads * embed_dim)]
        y = rearrange(y, "b h t d -> b t (h d)")
        y = block.resid_dropout(block.att_out_proj(y))

        # Include input residual.
        x += y

        # Apply FFN
        x += block.ffn(block.ln_2(x))

    # Final layernorm.
    x = model.layernorm(x)

    # Predict distribution over next tokens.
    return model.lm_head(x)
```

## Training

In this section, I train the transformer model from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. You can also skip this training part and just use the model I trained and included on the [github repo](https://github.com/ASzot/kv-cache/blob/main/model.pth). The training code automatically downloads the TinyStories dataset and trains for 1 epoch over the dataset. Thue model trains on subsequences of 256 tokens, which can span multiple stories.
```python
max_ctx_len = 256
tokenizer = tiktoken.get_encoding("gpt2")

device = torch.device("cuda:0")
model = CausalTransformer(
    vocab_size=tokenizer.n_vocab,
    embed_dim=384,
    dropout=0.2,
    n_layers=6,
    n_heads=6,
    max_ctx_len=max_ctx_len,
).to(device)
for block in model.blocks:
    block.mask = block.mask.to(device)
```
This model configuration has around 30 million parameters. Training took around 2 hours on a single A40 GPU using a batch size of 128 producing the following loss curve. 

{{< image src="/img/kv_cache/loss.png" class="center-image" width="600px" caption="Loss curve training the transformer model on the TinyStories dataset for 1 epoch." >}}

The final trained model produces reasonable text generations like the following (with the prompt in bold):

"_**Once upon a time, there was** a boy named Tim. Tim loved to play with his toys and run around outside. One day, Tim's mom told him to be careful and not go too far. Tim didn't listen and kept playing with his toys._
_As Tim was playing, he saw a big, scary dog. The dog was barking and running away. Tim's mom told him to be careful and not go near the dog. Tim listened to his mom and told her about the dog._
_His mom listened and listened to the dog. Tim was happy to have a new friend and played with his toys. From that day on, Tim learned to be careful and not to touch things._"

## Generation Without KV Cache
The simplest way to do inference is reuse the `transformer_forward` call to generate each new token. The current sequence of $n$ tokens is input to `transformer_forward` which outputs a $n$ next token probability distributions. We take the most likely next token for the final next token prediction and then add this to the list of tokens, resulting in $n+1$ tokens. We repeat this process to generate the entire response.
```python
def generate_text(
    prompt: str,
    n_gen_toks: int,
    model: CausalTransformer,
    stop_on_end_token: bool = False,
):
    # Tokenize the prompt and add a batch dimension of 1.
    prefix = torch.tensor(tokenizer.encode(prompt)).to(device).view(1, -1)

    # Output this text to the stdout.
    stream_text(prompt)

    # Sequence of current generated tokens.
    cur_toks = prefix
    for _ in range(n_gen_toks):

        # Compute the distribution over next tokens for all tokens.
        next_token_probs = transformer_forward(model, cur_toks)

        # Predict the most likely next token.
        next_tok = next_token_probs[:, -1:].argmax(-1)

        # Add the predicted token to the generated tokens.
        cur_toks = torch.cat([cur_toks, next_tok], dim=1)

        # Get the text for this new token.
        next_s = tokenizer.decode([next_tok.item()])
        if stop_on_end_token and next_s == END_TOKEN:
            break
        # Stream the newly generated text to stdout.
        stream_text(next_s)
```
Time how long it takes to generate 500 tokens. 
```python
prompt = "Once upon a time, there was "
num_test_tokens = 500

# Time how long generation takes when not using any KV cache.
with torch.no_grad():
    start = time.time()
    generate_text(prompt, num_test_tokens, model)
    print(f"\n# Tokens Per Second = {num_test_tokens / (time.time() - start)}\n")
```
This achieves **27** tokes per second.

## KV Cache - Dynamic
We can significantly improve on the 27 tokens per-second by using a KV cache. We implement the KV cache by providing an alternative forward pass implementation that will be used for text generation. This new forward pass produces the _exact same outputs_ as `transformer_forward`, but does so more efficiently by reusing the KV activations as described in detail in the previous [KV cache formulation](#kv-cache-formulation) section. The new forward pass returns the next token distributions along with the KV cache.

Now the `idx` inputs will be length $n$ for an $n$ token prompt and then length $1$ for every subsequent call. We need to adjust the position offset by the number of tokens in the KV cache. After processing $k$ tokens, the attention matrix is now rectangular with shape $1 \times (k+1)$ for every call after processing the prompt. The $1$ is because only a single token is processed at a time.
```python
def transformer_forward_kv(model: CausalTransformer, idx: Tensor, kv_cache: Tensor):
    device = idx.device
    batch_size, ctx_len = idx.shape

    # `kv_offset` tracks how many elements are already in the KV cache.
    kv_offset = 0
    if kv_cache is not None:
        kv_offset = kv_cache.shape[-2]

    # Embed input IDs and include position information. Account for the
    # existing elements in the KV cache.
    pos_embed = model.pos_embed(
        torch.arange(kv_offset, kv_offset + ctx_len, dtype=int, device=device)
    )
    x = model.tok_embed(idx) + pos_embed
    x = model.input_dropout(x)

    # Tensor that stores the KV cache for the new tokens.
    # Later combine this KV cache for new tokens with the existing `kv_cache`
    # for old tokens.
    # KV cache shape [#layers, 2, batch_size, #heads, context_len, embed_dim]
    new_kv_cache = torch.zeros(
        (
            len(model.blocks),
            # 2 because we need to fit both K and V tensors.
            2,
            # Batch size 1
            batch_size,
            model.n_heads,
            ctx_len,
            model.embed_dim // model.n_heads,
        ),
        device=device,
    )

    # Apply attention and FFN layers.
    for layer_idx, block in enumerate(model.blocks):
        q, k, v = block.qkv_proj(block.ln_1(x)).split(model.embed_dim, dim=2)

        q = rearrange(q, "b t (h k) -> b h t k", h=model.n_heads)
        k = rearrange(k, "b t (h k) -> b h t k", h=model.n_heads)
        v = rearrange(v, "b t (h v) -> b h t v", h=model.n_heads)

        # Save the calculated K, V values.
        new_kv_cache[layer_idx, 0] = k
        new_kv_cache[layer_idx, 1] = v

        if kv_cache is not None:
            # Add the previously calculated K,V values for computing the
            # attention.
            # Join on the sequence dimension.
            k = torch.cat([kv_cache[layer_idx, 0], k], -2)
            v = torch.cat([kv_cache[layer_idx, 1], v], -2)

        att = (q @ k.transpose(-2, -1)) / np.sqrt(k.shape[-1])
        # `att` is shape [batch_size, num_heads, ctx_len, #total_tokens]
        # where #total_tokens refers to the total number of tokens including in
        # the KV cache. Note that ctx_len will be 1 except when processing the
        # prompt.

        # Mask out the upper triangular part of the matrix for causal self-attention.
        # Assign to `-inf` so the softmax sets it to 0.
        # For the KV cache, we must account that we are currently calculating
        # the attention scores for `ctx_len` tokens (which is 1 most of the
        # time) but we have `kv_offset` tokens already in the KV cache.
        att = att.masked_fill(
            block.mask[:, :, kv_offset : kv_offset + ctx_len, : kv_offset + ctx_len]
            == 0,
            float("-inf"),
        )
        # `att` is still shape [batch_size, num_heads, ctx_len, #total_tokens]

        # Softmax per row.
        att = F.softmax(att, dim=-1)
        att = block.attn_dropout(att)

        y = att @ v
        y = rearrange(y, "b h t d -> b t (h d)")
        y = block.resid_dropout(block.att_out_proj(y))

        # `y` is `x` after self-attention operator.
        x += y
        x += block.ffn(block.ln_2(x))

    x = model.layernorm(x)
    if kv_cache is not None:
        # Combine the new KV cache tokens with the existing KV cache.
        # Join on the sequence dimension.
        new_kv_cache = torch.cat([kv_cache, new_kv_cache], -2)

    return model.lm_head(x), new_kv_cache
```
Timing it with the same setup gets **248** tokens per second, a **9.2x** speedup over no KV cache. However, this KV cache implementation uses a _dynamically allocated_ KV cache since we recreate the `new_kv_cache` tensor on every forward pass call to account for the new token. Reallocating this tensor on every forward pass is inefficient. We fix this issue in the next section.

## KV Cache - Preallocated

A more efficient KV cache implementation preallocates the KV cache tensor ahead of time. Now the code writes the new $K, V$ tensors to the KV cache tensor in place during each generation step. The attention matrix is now $1 \times N$ where $N$ is the _maximum sequence length_$. The causal attention mask ensures that the token we are currently predicting does not attend to future KV cache positions (which also are not yet initialized).
```python
def transformer_forward_kv_preallocated(
    model: CausalTransformer,
    input_tokens: Tensor,
    kv_cache: Tensor,
    seq_idxs: Tensor,
    pos_embed: Tensor,
):
    """
    :param kv_cache: This tensor has shape
        (#layers, 2, batch_size, #heads, max_context_length, embed_dim)
        Write to this tensor in place.
    :param seq_idxs: Sequence indices of the current input `idxs`.
    """

    # Embed input IDs and include position information.
    x = model.tok_embed(input_tokens) + pos_embed[seq_idxs]
    x = model.input_dropout(x)

    # Apply self-attention and feedforward layers.
    for layer_idx, block in enumerate(model.blocks):
        q, k, v = block.qkv_proj(block.ln_1(x)).split(model.embed_dim, dim=2)

        q = rearrange(q, "b t (h k) -> b h t k", h=model.n_heads)
        k = rearrange(k, "b t (h k) -> b h t k", h=model.n_heads)
        v = rearrange(v, "b t (h v) -> b h t v", h=model.n_heads)

        # Write the new KV values into the cache.
        # `kv_cache` has shape:
        # [#layers, 2, batch_size, #heads, max_ctx_len, hidden_dim]
        kv_cache[layer_idx, 0, :, :, seq_idxs] = k
        kv_cache[layer_idx, 1, :, :, seq_idxs] = v

        # Get the K, V values for the current layer.
        k = kv_cache[layer_idx, 0]
        v = kv_cache[layer_idx, 1]
        # k,v are shape: [batch_size, #heads, max_context_length, hidden_dim]

        att = (q @ k.transpose(-2, -1)) / np.sqrt(k.shape[-1])
        # `att` is shape [batch_size, #heads, #input tokens, max_context_length]

        # Mask out the upper triangular part of the matrix for causal self-attention.
        # Assign to `-inf` so the softmax sets it to 0.
        att = att.masked_fill(
            block.mask[:, :, seq_idxs] == 0,
            float("-inf"),
        )
        # `att` is still shape [batch_size, #heads, #input tokens, max_context_length]

        # Softmax per row.
        att = F.softmax(att, dim=-1)
        att = block.attn_dropout(att)

        y = att @ v
        y = rearrange(y, "b h t d -> b t (h d)")
        y = block.resid_dropout(block.att_out_proj(y))

        # `y` is `x` after self-attention operator.
        x += y
        x += block.ffn(block.ln_2(x))

    x = model.layernorm(x)

    return model.lm_head(x)
```
Timing this approach gets **327** tokens per second a **1.3x** speedup over using the dynamically allocated KV cache. The preallocated KV cache is also compatible with the `torch.compile` optimization because the input shapes to `transformer_forward_kv_preallocated` are constant. The below code shows both with and without compiling the forward pass. The forward pass for generating the first token will have a different `input_tokens` shape than subsequent calls since it must process the prompt. Therefore, the code gives 2 warmup steps to compile the forward function for single token generation.
```python
def generate_text_kv_preallocated(
    prompt: str,
    n_gen_toks: int,
    model: CausalTransformer,
    should_compile: bool,
    stop_on_end_token: bool = False,
):
    prefix = torch.tensor(tokenizer.encode(prompt)).to(device).view(1, -1)
    stream_text(prompt)
    cur_toks = prefix

    # Pre-allocate a certain KV cache size.
    kv_cache = torch.zeros(
        (
            len(model.blocks),
            # 2 because we need to fit both K and V tensors.
            2,
            # Batch size 1
            1,
            model.n_heads,
            max_ctx_len,
            model.embed_dim // model.n_heads,
        ),
        device=device,
    )

    # Preallocate all position embeddings.
    pos_embed = model.pos_embed(torch.arange(0, max_ctx_len, dtype=int, device=device))

    seq_idxs = torch.arange(max_ctx_len).to(device)
    if should_compile:
        global transformer_forward_kv_preallocated
        forward_fn = torch.compile(transformer_forward_kv_preallocated)
    else:
        forward_fn = transformer_forward_kv_preallocated

    # Track the current starting token sequence index
    tok_idx = 0

    # 1 step for the prompt (which has a different shape) and 1 step for the
    # 1st generated token. Then all the shapes are consistent.
    warmup_steps = 2

    for i in range(n_gen_toks + warmup_steps):
        if i == warmup_steps:
            # Only start profiling after `warmup_steps`. 
            start = time.time()

        ctx_len = cur_toks.shape[1]
        pred = forward_fn(
            model,
            cur_toks,
            kv_cache,
            seq_idxs[tok_idx : tok_idx + ctx_len],
            pos_embed,
        )
        tok_idx += ctx_len

        # Predict the most likely next token.
        next_tok = pred[:, -1:].argmax(-1)
        cur_toks = next_tok

        next_s = tokenizer.decode([next_tok.item()])
        if stop_on_end_token and next_s == END_TOKEN:
            break
        stream_text(next_s)
```
Running with torch compile results in **554** tokens per second. This is a **1.7x** speedup over the non-compiled version.

## Conclusion

The final KV cache implementation with the preallocatd KV cache and torch compile was almost **21x** faster than the generation code without any KV cache. The numbers per setting are summarized below.

| Setting | Steps-Per-Second | 
|------|:------:|
|   No KV Cache |   27   |
|   Dynamic KV Cache |   248 |
|   Preallocated KV cache |   327 |
|   Preallocated KV cache - with torch compile|   554   |

All the code for this post is at [github.com/ASzot/kv-cache](https://github.com/ASzot/kv-cache). The code is just a short single Python file and only takes a couple seconds to run on a consumer grade GPU.

If you have any questions or spot any errors, contact [asz.post.contact@gmail.com](mailto:asz.post.contact@gmail.com). Join [the email list here](https://mailchi.mp/30a660245978/add-email) to be notified about new posts.

## Changes
- July 8th, 2024: Posted.
