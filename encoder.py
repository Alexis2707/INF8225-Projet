import torch
import torch.nn as nn
from attention import MultiheadAttention # Assuming attention.py is in the same directory

class TransformerEncoderLayer(nn.Module):
    """Single encoder layer.

    Parameters:
        d_model: The dimension of input tokens.
        d_ff: Hidden dimension of the feedforward networks.
        nhead: Number of heads for each multi-head attention.
        dropout: Dropout rate.
        device: Device to run the layer on.
    """
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            nhead: int,
            dropout: float,
            device: str = 'cpu',
        ):
        super().__init__()
        self.device = device
        self.self_attn = MultiheadAttention(
            dim=d_model,
            n_heads=nhead,
            dropout=dropout,
            device=self.device # Pass device
            )

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff, device=self.device),
            nn.ReLU(),
            nn.Dropout(dropout), # Add dropout in FF layer too
            nn.Linear(d_ff, d_model, device=self.device)
        )

        self.norm1 = nn.LayerNorm(d_model, device=self.device)
        self.norm2 = nn.LayerNorm(d_model, device=self.device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.FloatTensor,
        src_key_padding_mask: torch.BoolTensor = None # Make mask optional
        ) -> torch.FloatTensor:
        """
        Args:
            src: Batch of embedded source tokens [batch_size, src_seq_len, dim_model].
            src_key_padding_mask: Mask preventing attention to padding tokens [batch_size, src_seq_len]. True indicates padding.

        Returns:
            Batch of encoded source tokens [batch_size, src_seq_len, dim_model].
        """
        # Self Attention Block
        # Note: Q, K, V are all src for self-attention
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output) # Residual connection
        src = self.norm1(src) # Layer Norm

        # Feed Forward Block
        ff_output = self.feedforward(src)
        src = src + self.dropout2(ff_output) # Residual connection
        src = self.norm2(src) # Layer Norm

        return src


class TransformerEncoder(nn.Module):
    """Transformer encoder stack.

    Parameters:
        d_model: The dimension of encoders inputs.
        d_ff: Hidden dimension of the feedforward networks.
        num_encoder_layers: Number of stacked encoders.
        nhead: Number of heads for each multi-head attention.
        dropout: Dropout rate.
        device: Device to run the layers on.
    """
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_encoder_layers: int,
            nhead: int,
            dropout: float,
            device: str = 'cpu',
        ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, d_ff, nhead, dropout, device=self.device)
            for _ in range(num_encoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model, device=self.device) # Optional final norm

    def forward(
            self,
            src: torch.FloatTensor,
            src_key_padding_mask: torch.BoolTensor = None # Make mask optional
        ) -> torch.FloatTensor:
        """
        Args:
            src: Batch of embedded source sentences [batch_size, src_seq_len, dim_model].
            src_key_padding_mask: Mask preventing attention to padding tokens [batch_size, src_seq_len]. True indicates padding.

        Returns:
            Batch of encoded source sequence [batch_size, src_seq_len, dim_model].
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)

        output = self.norm(output) # Apply final norm if added
        return output
