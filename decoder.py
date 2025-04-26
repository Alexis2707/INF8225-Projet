import torch
import torch.nn as nn
from attention import MultiheadAttention # Assuming attention.py is in the same directory

class TransformerDecoderLayer(nn.Module):
    """Single decoder layer.

    Parameters:
        d_model: Dimension of inputs/outputs.
        d_ff: Hidden dimension of the feedforward networks.
        nhead: Number of heads for multi-head attention.
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
            dim=d_model, n_heads=nhead, dropout=dropout, device=self.device
        )
        self.cross_attn = MultiheadAttention(
            dim=d_model, n_heads=nhead, dropout=dropout, device=self.device
        )

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff, device=self.device),
            nn.ReLU(),
            nn.Dropout(dropout), # Add dropout in FF layer too
            nn.Linear(d_ff, d_model, device=self.device)
        )

        self.norm1 = nn.LayerNorm(d_model, device=self.device)
        self.norm2 = nn.LayerNorm(d_model, device=self.device)
        self.norm3 = nn.LayerNorm(d_model, device=self.device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
            self,
            tgt: torch.FloatTensor,
            memory: torch.FloatTensor, # Renamed src to memory for clarity
            tgt_mask: torch.BoolTensor = None, # Causal mask [tgt_len, tgt_len]
            memory_mask: torch.BoolTensor = None, # Not typically used here, pass None
            tgt_key_padding_mask: torch.BoolTensor = None, # [b, tgt_len]
            memory_key_padding_mask: torch.BoolTensor = None, # [b, src_len]
        ) -> torch.FloatTensor:
        """
        Args:
            tgt: Batch of target sequences [batch_size, tgt_seq_len, dim_model].
            memory: Output from the encoder [batch_size, src_seq_len, dim_model].
            tgt_mask: Mask to prevent attention to subsequent tokens [tgt_seq_len, tgt_seq_len]. True indicates masking.
            memory_mask: Not used in standard Transformer decoder layer.
            tgt_key_padding_mask: Mask for padding in target sequence [batch_size, tgt_seq_len]. True indicates padding.
            memory_key_padding_mask: Mask for padding in memory (encoder output) [batch_size, src_seq_len]. True indicates padding.

        Returns:
            Output tensor [batch_size, tgt_seq_len, dim_model].
        """
        # Masked Self-Attention Block
        self_attn_output, _ = self.self_attn(
            q_in=tgt, k_in=tgt, v_in=tgt,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_mask # Causal mask
        )
        tgt = tgt + self.dropout1(self_attn_output) # Residual connection
        tgt = self.norm1(tgt) # Layer Norm

        # Cross-Attention Block (Query: target, Key/Value: memory from encoder)
        cross_attn_output, _ = self.cross_attn(
            q_in=tgt, k_in=memory, v_in=memory,
            key_padding_mask=memory_key_padding_mask # Padding mask for encoder output
        )
        tgt = tgt + self.dropout2(cross_attn_output) # Residual connection
        tgt = self.norm2(tgt) # Layer Norm

        # Feed Forward Block
        ff_output = self.feedforward(tgt)
        tgt = tgt + self.dropout3(ff_output) # Residual connection
        tgt = self.norm3(tgt) # Layer Norm

        return tgt

class TransformerDecoder(nn.Module):
    """Transformer decoder stack.

    Parameters:
        d_model: Dimension of inputs/outputs.
        d_ff: Hidden dimension of the feedforward networks.
        num_decoder_layers: Number of stacked decoders.
        nhead: Number of heads for multi-head attention.
        dropout: Dropout rate.
        device: Device to run the layers on.
    """
    def __init__(
            self,
            d_model: int,
            d_ff: int,
            num_decoder_layers: int,
            nhead: int,
            dropout: float,
            device: str = 'cpu',
        ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, d_ff, nhead, dropout, device=self.device)
            for _ in range(num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model, device=self.device) # Optional final norm

    def forward(
            self,
            tgt: torch.FloatTensor,
            memory: torch.FloatTensor,
            tgt_mask: torch.BoolTensor = None,
            memory_mask: torch.BoolTensor = None, # Not used
            tgt_key_padding_mask: torch.BoolTensor = None,
            memory_key_padding_mask: torch.BoolTensor = None,
        ) -> torch.FloatTensor:
        """
        Args:
            See TransformerDecoderLayer forward method args. Memory is encoder output.

        Returns:
            Final output tensor [batch_size, tgt_seq_len, dim_model].
        """
        output = tgt
        for layer in self.layers:
            output = layer(
                tgt=output,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask, # Pass None
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        output = self.norm(output) # Apply final norm if added
        return output
