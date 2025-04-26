import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from einops.layers.torch import Rearrange


def attention(
        q: torch.FloatTensor,
        k: torch.FloatTensor,
        v: torch.FloatTensor,
        mask: torch.BoolTensor=None,
        dropout: nn.Dropout=None,
    ) -> tuple:
    """Computes multihead scaled dot-product attention.

    Args:
        q: Queries [batch_size, n_heads, seq_len_1, dim_head].
        k: Keys [batch_size, n_heads, seq_len_2, dim_head].
        v: Values [batch_size, n_heads, seq_len_2, dim_head].
        mask: Attention mask [batch_size, n_heads, seq_len_1, seq_len_2] or broadcastable. True values are masked.
        dropout: Dropout layer.

    Returns:
        Tuple of (output [batch_size, n_heads, seq_len_1, dim_head], attention_weights [batch_size, n_heads, seq_len_1, seq_len_2]).
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # [b, h, s1, s2]

    if mask is not None:
         # Ensure mask is boolean and correctly broadcastable
         # Mask needs to be [b, h, s1, s2] or broadcastabe to it.
         # Common masks might be [b, 1, 1, s2] (key_padding) or [1, 1, s1, s2] (causal)
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    output = torch.matmul(attn_weights, v) # [b, h, s1, dh]
    return output, attn_weights

class MultiheadAttention(nn.Module):
    """Multihead attention module.

    Parameters:
        dim: Dimension of the input tokens.
        n_heads: Number of heads. `dim` must be divisible by `n_heads`.
        dropout: Dropout rate.
    """
    def __init__(
            self,
            dim: int,
            n_heads: int,
            dropout: float,
            device='cpu' # Add device parameter
        ):
        super().__init__()
        assert dim % n_heads == 0
        self.device = device
        self.n_heads = n_heads
        self.dim_head = dim // n_heads

        # Use device parameter for linear layers
        self.w_q = nn.Linear(dim, dim, bias=False, device=self.device)
        self.w_k = nn.Linear(dim, dim, bias=False, device=self.device)
        self.w_v = nn.Linear(dim, dim, bias=False, device=self.device)
        self.w_o = nn.Linear(dim, dim, bias=False, device=self.device)

        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            q_in: torch.FloatTensor, # Rename input args to avoid clash with projected ones
            k_in: torch.FloatTensor,
            v_in: torch.FloatTensor,
            key_padding_mask: torch.BoolTensor = None, # [b, s2]
            attn_mask: torch.BoolTensor = None, # [s1, s2] or [b, s1, s2]
        ) -> tuple: # Return attention weights as well
        """
        Args:
            q_in: Query input [batch_size, seq_len_1, dim_model].
            k_in: Key input [batch_size, seq_len_2, dim_model].
            v_in: Value input [batch_size, seq_len_2, dim_model].
            key_padding_mask: Mask for padding in keys/values [batch_size, seq_len_2]. True indicates padding.
            attn_mask: Additional attention mask (e.g., causal) [seq_len_1, seq_len_2]. True indicates masking.

        Returns:
            Tuple of (output [batch_size, seq_len_1, dim_model], attention_weights [batch_size, n_heads, seq_len_1, seq_len_2]).
        """
        batch_size, seq_len_1, _ = q_in.shape
        _, seq_len_2, _ = k_in.shape

        # 1. Linear projections
        q = self.w_q(q_in)
        k = self.w_k(k_in)
        v = self.w_v(v_in)

        # 2. Reshape for multi-head attention
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.n_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.n_heads)

        # 3. Combine masks
        # key_padding_mask: [b, s2] -> [b, 1, 1, s2]
        # attn_mask: [s1, s2] -> [1, 1, s1, s2]
        combined_mask = None
        if key_padding_mask is not None:
            combined_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # [b, 1, 1, s2]
            # Ensure mask is boolean
            combined_mask = combined_mask.bool()

        if attn_mask is not None:
            attn_mask_unsqueezed = attn_mask.unsqueeze(0).unsqueeze(1) # [1, 1, s1, s2]
             # Ensure mask is boolean
            attn_mask_unsqueezed = attn_mask_unsqueezed.bool()
            if combined_mask is not None:
                # Combine by logical OR: mask if either mask is True
                combined_mask = combined_mask | attn_mask_unsqueezed # Broadcasts to [b, 1, s1, s2]
            else:
                combined_mask = attn_mask_unsqueezed

        # 4. Apply attention
        # The mask needs to be [b, h, s1, s2] for the attention function
        # combined_mask is currently [b, 1, s1, s2] or [1, 1, s1, s2] or [b, 1, 1, s2]
        # It will broadcast correctly if dimensions match or are 1.
        y, attn_weights = attention(q, k, v, mask=combined_mask, dropout=self.dropout)

        # 5. Concatenate heads and final linear layer
        y = rearrange(y, "b h s d -> b s (h d)")
        output = self.w_o(y)

        return output, attn_weights # Return both output and weights
