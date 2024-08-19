---
title: Position Embeddings
date: 2024-07-24
---


## Intro

Transformer models need position embeddings to distinguish the ordering of the inputs. The choice of position embedding is important for a transformer to process longer sequences of inputs than it was trained on. This post explores several position popular embedding methods. The implementations are minimal, self-contained and in pure PyTorch. Experiments use the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset, meaning models only take a couple hours to train from scratch on a consumer grade GPU. The code is at [github.com/aszot/position-embeddings](https://github.com/ASzot/position-embeddings) and is only a small single Python file.

## General Framework

This section formally defines the transformer layer and how position embeddings modify it.

Let $X \in \mathbb{R}^{N \times D} $ be the input embeddings, $ W^{Q}, W^{K}, W^{V} $ be the $ D \times D$ projection matrices for the query, key, value and output respectively and $ \text{FFN}$ to be a 2-layer MLP. Then a standard transformer layer computes:
$$
X_{\text{ln}} = \text{LayerNorm}(X), K = X_{\text{ln}} W^K, Q = X_{\text{ln}} W^Q, V = X_{\text{ln}} W^V \\\
X_{\text{Att}} = X_{\text{ln}} + \text{Softmax} \left( \frac{QK^\top}{\sqrt{D}} \right) V \\\
\text{Output} = X_{\text{Att}} + \text{FFN} \left( \text{LayerNorm}\left( X_{\text{Att}} \right)  \right) 
$$
The position embedding methods this post explores inject position into the transformer layer in one of two ways:

1. **Input Embeddings:** A $N \times D$ tensor is added to only the first transformer layer to incorporate position information.

2. **Modifying Attention**: The $\text{Softmax} \left( \frac{QK^\top}{\sqrt{D}} \right)$ is modified to incorporate position information, typically through modeling some pairwise relation between elements in the input sequence. 

These position embeddings can be summarized in the following interface:
```python
class PositionEmbedding(nn.Module):
    def get_attn_mask(self, ctx_len: int, device) -> Tensor:
        """
        Return attention mask used for attention.
        :returns: Broadcastable to shape [batch_size, num_heads, context_len,
            context_len].
        """

        # By default get a causal attention mask.
        return torch.tril(
            torch.ones((ctx_len, ctx_len), device=device, dtype=torch.bool)
        ).view(1, 1, ctx_len, ctx_len)

    def get_pos_embed_offset(self, idxs: Tensor) -> Tensor:
        """
        :params idxs: Shape [batch_size, context_len] tensor of indices of the
            transformer input.
        :returns: Shape [batch_size, context_len, embed_dim] tensor to be added to
            the input embeddings tensors.
        """

    def modify_qk(self, q_or_v: Tensor) -> Tensor:
        """
        Modify the `Q` or `V` tensors.
        """

        return q_or_v
``` 
This interface is used in the following transformer forward pass.
```python
def transformer_forward(model: CausalTransformer, input_tokens: Tensor) -> Tensor:
    """
    :param input_tokens: Shape [batch_size, context_len] input tokens IDs.

    :returns: Shape [batch_size, context_len, #tokens] prediction over next
        token probabilities.
    """

    ctx_len = input_tokens.shape[1]
    device = input_tokens.device

    # Embed input IDs
    x = model.tok_embed(input_tokens)

    # `pos_embed` is shape [ctx_len, hidden_dim]
    pos_embed = model.position_embedder.get_pos_embed_offset(
        idxs=torch.arange(0, ctx_len, dtype=int, device=input_tokens.device)
    )
    if pos_embed is not None:
        # Include positional information.
        x = x + pos_embed.to(input_tokens.device)

    # `attn_mask` is shape [1, #heads (or 1), ctx_len, ctx_len]
    attn_mask = model.position_embedder.get_attn_mask(ctx_len, device)

    for i, block in enumerate(model.blocks):
        q, k, v = block.qkv_proj(block.att_norm(x)).split(model.embed_dim, dim=2)

        # Convert Q,K,V to shape [batch_size, #heads, ctx_len, embed_dim]
        q = rearrange(q, "b t (h k) -> b h t k", h=model.n_heads)
        k = rearrange(k, "b t (h k) -> b h t k", h=model.n_heads)
        v = rearrange(v, "b t (h v) -> b h t v", h=model.n_heads)

        # Possibly incorporate position information.
        q = model.position_embedder.modify_qk(q)
        k = model.position_embedder.modify_qk(k)

        # Apply attention. Unless changed by the position embedding,
        # `attn_mask` is causal.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Final shape output of attention operator is
        # [batch_size, ctx_len, (#heads * embed_dim)]
        y = rearrange(y, "b h t d -> b t (h d)")
        y = block.att_out_proj(y)

        # Att-block residual connection
        x = x + y

        # Apply FFN
        x = x + block.ffn(block.ffn_norm(x))

    # Predict distribution over next tokens.
    return model.lm_head(model.output_norm(x))
```

## Absolute Position Embedding
In ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), absolute position embeddings are the simplest position embedding approach where embeddings are learned end-to-end for each context length position. Specifically, the input embedding to the first transformer layer $X^0 \in \mathbb{R}^{N \times D}$ is modified via $ X + E$ where $ E \in \mathbb{R}^{N \times D}$ is a randomly initialized learned tensor. $E$ is learned end-to-end with the rest of the transformer. Using the `PositionEmbedding` interface, this is implemented as: 
```python
class AbsoluteEmbedding(PositionEmbedding):
    def __init__(self, max_context_len: int, embed_dim: int, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(max_context_len, embed_dim)

    def get_pos_embed_offset(self, idxs):
        return self.embed(idxs)
```
The below plot shows the training loss when using the absolute position embedding to train a language model from scratch on the TinyStories dataset. The model is a [Llama-2](https://arxiv.org/abs/2307.09288) style transformer, meaning using RMSNorm and SwiGLU activations. The model has 6 layers 384 embedding dimension and is trained for 1 epoch over the dataset. Most importantly for this investigation, the model is **trained with context length 256**.

{{< image src="/img/position_embeddings/abs_train.png" class="center-image" width="600px" caption="Train loss with absolute position embedding." >}}

While the model can learn and operate over sequences of length less than $ 256$ it has no hope of extrapolating beyond 256 tokens because $E_n$ is still random for $n > 256$. The other position embedding methods try to give the model some extrapolation capabilities for beyond the training sequence length.



## Sinusoidal Position Embedding

Also in ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762), the sinusoidal position embedding combines the input embedding with a fixed offset computed with sine or cosine. The input embedding to the first transformer layer $X^0 \in \mathbb{R}^{N \times D}$ is again modified via $ X + E$ where $E \in \mathbb{R}^{N \times D}$ is a fixed matrix with, $E_{n,d} = \sin \left( \frac{n}{10,000} \right)^{ \frac{2d}{D}} $ for even $ d$ and $ \cos$ instead of $ \sin $ for odd $ d$. However, unlike the absolute position embedding, this matrix $E$ is fixed throughout training.
```python
class SinusoidalEmbedding(PositionEmbedding):
    def __init__(self, max_context_len: int, embed_dim: int, **kwargs):
        super().__init__()
        # The position embedding is constant and not learnable.
        pos_embeds = torch.zeros((max_context_len, embed_dim))

        # `pos_id` is shape [max_context_len, 1]
        pos_id = torch.arange(max_context_len).unsqueeze(1)

        # `pos_multiplier` is shape [embed_dim / 2,]
        pos_multiplier = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-np.log(10_000) / embed_dim)
        )

        # Even embedding indices. Compute an outer product to produce shape
        # (max_context_len, embed_dim/2)
        pos_embeds[:, 0::2] = torch.sin(pos_id * pos_multiplier)
        # Odd embedding indices
        pos_embeds[:, 1::2] = torch.cos(pos_id * pos_multiplier)
        self.register_buffer("pos_embeds", pos_embeds)

    def get_pos_embed_offset(self, idxs):
        return self.pos_embeds[: idxs.shape[0]]
```
The below plot shows the training loss when using this position embedding to train a language model in the same setting as above. 

{{< image src="/img/position_embeddings/sin_train.png" class="center-image" width="600px" caption="Train loss with sinusoidal position embedding." >}}

The hope is that the model will extrapolate to $E_{n}$ for $n $ greater than the training context length of 256. The following figure shows the loss on _train tokens_ for only the elements in context position between 256 and 512. Note that this is on the same training text, and thus only shows length extrapolation abilities. As the figure shows, the loss is very high for these longer sequences, meaning the model fails to extrapolate to these sequences with sinusoidal position embeddings.

{{< image src="/img/position_embeddings/sin_ext.png" class="center-image" width="600px" caption="Extrapolation to longer sequences with sinusoidal position embeddings with the plot displaying average loss on token positions 256-512." >}}

## Attention with Linear Biases (ALiBi)

[Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409) position embeddings seeks to enable context length extrapolation by adding a pairwise distance penalty in the attention computation. Specifically, the $N \times N$ attention scores $ QK^\top$ are modified by adding a fixed $N \times N$ offset tensor. Specifically, for the softmax input, we compute $QK^\top + E \cdot m$ where $E_{ij} = |i - j|$ when $ i > j$ or 0 otherwise (upper right triangle is all zeros, diagonal is all 0 and lower left triangle are relative distances) and $m $ is a per-attention head scaling factor. We can combine this operation with the causal attention mask which sets the upper right triangle of $ E$ to $ -\infty $. Like sinusoidal embeddings, ALiBi has no learned parameters. But unlike sinusoidal embeddings, AliBi is incorporated in every transformer layer, not just the input. AliBi also encodes relative position between sequence indices, rather than the absolute sequence index. In code:
```python
class ALiBiEmbedding(PositionEmbedding):
    def __init__(
        self, max_context_len: int, embed_dim: int, n_heads: int, device, **kwargs
    ):
        super().__init__()
        self.attn_mask = None

        # `index_distance` is shape [max_context_len, max_context_len]
        # `index_distance[i,j] = abs(j-i)` penalizes elements for being far away.
        index_distance = torch.arange(max_context_len).view(1, -1) - torch.arange(
            max_context_len
        ).view(-1, 1)

        start = 2 ** (-8 / n_heads)
        m = torch.tensor([start * (start) ** (i + 1) for i in range(n_heads)]).view(
            1, -1, 1, 1
        )
        # `mask` is shape [n_heads, max_context_len, max_context_len]
        mask = index_distance * m

        causal_mask = torch.triu(torch.ones_like(mask), diagonal=1).bool()
        # Make the mask causal by setting any future positions to -inf.
        # This mask is ultimately ADDED to the QK^T scores, so -inf is
        # needed to set the attention score to -inf before the softmax.
        self.mask = mask.masked_fill_(causal_mask, float("-inf")).to(device)

    def get_attn_mask(self, ctx_len, device):
        return self.mask[:, :, :ctx_len, :ctx_len]
```
Training the language model with this position embedding results in extrapolation to longer sequence lengths as shown in the below figure. 

{{< image src="/img/position_embeddings/alibi_ext.png" class="center-image" width="600px" caption="Extrapolation to longer sequences with ALiBi position embeddings with the plot displaying average loss on token positions 256-512." >}}

## Rotary Position Embedding (RoPE)

Used in many LLMs such as [Llama3](https://llama.meta.com) and [Gemma](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf), [Rotary Position Embedding (RoPE)](https://arxiv.org/abs/2104.09864) also relies on relative positions between sequence elements. RoPE is also implemented as a fixed product on the $Q, K$ tensors. The key equation to implement is equation 34 at the bottom of page 7 from the RoPE paper. The _rotary matrix_ $R_{\Theta, m}^d$ encodes relative distances and is set by fixed parameters  $ \Theta = \{ \theta_1, \dots, \theta_{d/2}\}$ where $\theta_i = 10,000^{-2i / d}$. We apply the RoPE transformation by computing $ \left( R_{\Theta, m}^d Q \right) \left( R_{\Theta, m}^d K \right)^\top $ in place of the standard $QK^\top$ in the attention operation. Since $R_{\Theta, m}^d$ is a sparse matrix Eq. 34 of the paper shows we can simplify it's matrix product as: 
$$
R_{\Theta, m}^d x = 
\begin{pmatrix} 
x_1 \cos m \theta_1 \\\
x_2 \cos m \theta_1 \\\
x_3 \cos m \theta_2 \\\
x_4 \cos m \theta_2 \\\
\vdots \\\
x_{d-1} \cos m \theta_{d/2} \\\
x_d \cos m \theta_{d/2} 
\end{pmatrix}
+
\begin{pmatrix} 
-x_2 \sin m \theta_1 \\\
x_1 \sin m \theta_1 \\\
-x_4 \sin m \theta_2 \\\
x_3 \sin m \theta_2 \\\
\vdots \\\
-x_{d} \sin m \theta_{d/2} \\\
x_{d-1} \sin m \theta_{d/2} 
\end{pmatrix}
$$
To implement this, we must compute $\cos m \theta_i$ for all possible sequence indices $m$ and $i \in [0, d/2]$.
```python
class RoPE_Embedding(PositionEmbedding):
    def __init__(
        self, max_context_len: int, embed_dim: int, n_heads: int, device, **kwargs
    ):
        super().__init__()
        head_embed_dim = embed_dim // n_heads
        # From Section 3.2.2 of RoFormer paper.
        # `theta` shape: [head_embed_dim/2]
        theta = 1.0 / (10_000 ** (torch.arange(0, head_embed_dim, 2) / head_embed_dim))

        m = torch.arange(max_context_len)

        # `m_theta` shape: [max_context_len, head_embed_dim//2]
        m_theta = torch.outer(m, theta)

        # `sincos_table` shape: [max_context_len, head_embed_dim//2, 2]
        # Last dimension gives cos or sin of value. This is needed to compute
        # Eq. 34 of the RoFormer paper.
        self.sincos_table = torch.stack(
            [torch.cos(m_theta), torch.sin(m_theta)], -1
        ).to(device)

    def modify_qk(self, x):
        ctx_len, embed_dim = x.shape[-2:]

        sincos_table = self.sincos_table[:ctx_len]
        sincos_table = rearrange(sincos_table, "c d s -> 1 1 c d s")
        cos_table = sincos_table[..., 1]
        sin_table = sincos_table[..., 1]

        x_split = x.view(*x.shape[:-1], -1, 2)

        # Note that Eq. 34 is 1 indexed, but the operations below are 0 indexed.
        # Compute even indices of the OUTPUT
        even_terms = (
            # Even indices of inputs multiplied by cos terms
            x_split[..., 0] * cos_table
            # Odd indices of inputs multiplied by sin terms.
            # Notice the negative sign in front of the odd terms.
            - x_split[..., 1] * sin_table
        )
        # Compute odd indices of the OUTPUT
        odd_terms = (
            # Odd indices of inputs multiplied by cos
            x_split[..., 1] * cos_table
            # Even indices of inputs multiplied by cos
            + x_split[..., 0] * sin_table
        )

        # Stack the even and odd terms to be ordered 0,1,2,3,... again
        return torch.stack([even_terms, odd_terms], -1).view(x.shape)
```
Like Alibi, RoPE also extrapolates to longer sequence lengths as shown in the figure below.
{{< image src="/img/position_embeddings/rope_ext.png" class="center-image" width="600px" caption="Loss on context longer sequences of training tokens with RoPE position embeddings. The loss is computed on tokens in context positions 256-512." >}}

## Conclusion
As shown in the figures below, sinusoidal, ALiBi and RoPE, achieve relatively similar performance on the training distribution of context lengths. However, for extended context lengths, sinusoidal fails. ALiBi slightly outperforms RoPE, but by a small margin. Figure 1 of ALiBi [paper](https://arxiv.org/abs/2108.12409) also shows better extrapolation than RoPE.

{{< image src="/img/position_embeddings/train.png" class="center-image" width="600px" caption="Loss with context length 0-512." >}}

{{< image src="/img/position_embeddings/ext.png" class="center-image" width="600px" caption="Extrapolation to longer sequences with the plot displaying average loss on token positions 256-512." >}}

If you have any questions or spot any errors, contact [asz.post.contact@gmail.com](mailto:asz.post.contact@gmail.com). Join [the email list here](https://mailchi.mp/30a660245978/add-email) to be notified about new posts.


