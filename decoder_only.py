import torch
import torch.nn as nn
import math
from attention import MultiheadAttention # Assuming attention.py is in the same directory
from transformer import PositionalEncoding # Assuming transformer.py is in the same directory

class DecoderOnlyLayer(nn.Module):
    """Single Decoder-Only layer with masked self-attention.

    Parameters:
        d_model: Dimension of input tokens.
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
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff, device=self.device),
            nn.ReLU(),
            nn.Dropout(dropout), # Add dropout
            nn.Linear(d_ff, d_model, device=self.device)
        )
        self.norm1 = nn.LayerNorm(d_model, device=self.device)
        self.norm2 = nn.LayerNorm(d_model, device=self.device)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        seq: torch.FloatTensor, # [b, seq_len, dim]
        attn_mask: torch.BoolTensor = None, # [seq_len, seq_len] or broadcastable
        key_padding_mask: torch.BoolTensor = None # [b, seq_len]
        ) -> torch.FloatTensor:
        """
        Args:
            seq: Batch of embedded input tokens [batch_size, seq_len, dim_model].
            attn_mask: Causal mask preventing attention to future tokens [seq_len, seq_len]. True masks position.
            key_padding_mask: Mask preventing attention to padding tokens [batch_size, seq_len]. True masks position.

        Returns:
            Output tensor [batch_size, seq_len, dim_model].
        """
        # Masked Self-Attention Block
        attn_output, _ = self.self_attn(
            q_in=seq, k_in=seq, v_in=seq,
            attn_mask=attn_mask, # Causal mask
            key_padding_mask=key_padding_mask # Padding mask
        )
        seq = seq + self.dropout1(attn_output) # Residual connection
        seq = self.norm1(seq) # Layer Norm

        # Feed Forward Block
        ff_output = self.feedforward(seq)
        seq = seq + self.dropout2(ff_output) # Residual connection
        seq = self.norm2(seq) # Layer Norm

        return seq


class DecoderOnlyTransformer(nn.Module):
    """Stack of Decoder-Only layers.

    Parameters:
        d_model: Dimension of input/output embeddings.
        d_ff: Hidden dimension of the feedforward networks.
        nhead: Number of heads for multi-head attention.
        num_decoder_layers: Number of stacked decoder layers.
        dropout: Dropout rate.
        device: Device to run the layers on.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_decoder_layers: int,
        d_ff: int,
        dropout: float,
        device: str = 'cpu',
    ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([
            DecoderOnlyLayer(d_model, d_ff, nhead, dropout, device=self.device)
            for _ in range(num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(d_model, device=self.device) # Final normalization

    def forward(
        self,
        seq: torch.FloatTensor, # [b, seq_len, dim]
        attn_mask: torch.BoolTensor = None, # [seq_len, seq_len]
        key_padding_mask: torch.BoolTensor = None # [b, seq_len]
    ) -> torch.FloatTensor:
        """Processes the sequence through the stack."""
        output = seq
        for layer in self.layers:
            output = layer(
                seq=output,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask
            )
        output = self.norm(output) # Apply final norm
        return output


class DecoderOnlyTranslationTransformer(nn.Module):
    """Decoder-Only Transformer for sequence generation/translation.
       Assumes source and target are concatenated for input.

    Parameters:
        n_tokens_vocab: Size of the combined vocabulary.
        n_heads: Number of attention heads.
        dim_embedding: Dimension of token embeddings.
        dim_hidden: Dimension of the feedforward layers.
        num_layers: Number of decoder layers.
        dropout: Dropout rate.
        pad_idx: Index of the padding token.
        device: Device to run the model on.
    """
    def __init__(
        self,
        n_tokens_vocab: int, # Combined vocab size if src/tgt are merged
        n_heads: int,
        dim_embedding: int,
        dim_hidden: int,
        num_layers: int,
        dropout: float,
        pad_idx: int,
        device: str = 'cpu',
    ):
        super().__init__()
        self.device = device
        self.pad_idx = pad_idx

        self.token_embedding = nn.Embedding(n_tokens_vocab, dim_embedding, padding_idx=pad_idx, device=self.device)
        self.position_embedding = PositionalEncoding(dim_embedding, dropout, device=self.device)

        self.transformer = DecoderOnlyTransformer(
            d_model=dim_embedding,
            nhead=n_heads,
            num_decoder_layers=num_layers,
            d_ff=dim_hidden,
            dropout=dropout,
            device=self.device
        )

        self.final_out = nn.Linear(dim_embedding, n_tokens_vocab, device=self.device)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        sequence: torch.LongTensor # Concatenated source+target or just target [b, seq_len]
    ) -> torch.FloatTensor:
        """Predicts the next token logits for each position in the sequence."""

        key_padding_mask = self.generate_key_padding_mask(sequence, self.pad_idx)
        attn_mask = self.generate_causal_mask(sequence.size(1)) # Causal mask

        # Embed and add positional encoding
        seq_emb = self.position_embedding(self.token_embedding(sequence) * math.sqrt(self.transformer.layers[0].self_attn.dim_head * self.transformer.layers[0].self_attn.n_heads))

        # Pass through Decoder-Only Transformer stack
        processed_seq = self.transformer(
            seq=seq_emb,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask
        )

        # Final linear layer to get logits
        logits = self.final_out(processed_seq)
        return logits

    def generate_causal_mask(self, size: int) -> torch.BoolTensor:
        """Generates an upper-triangular matrix for causal masking."""
        mask = torch.triu(torch.ones(size, size, device=self.device, dtype=torch.bool), diagonal=1)
        return mask

    def generate_key_padding_mask(self, tokens: torch.LongTensor, pad_idx: int) -> torch.BoolTensor:
        """Generates a mask for padding tokens."""
        mask = (tokens == pad_idx)
        return mask
