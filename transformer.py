import torch
import torch.nn as nn
import math
from einops import rearrange
from encoder import TransformerEncoder # Assuming encoder.py is in the same directory
from decoder import TransformerDecoder # Assuming decoder.py is in the same directory

class PositionalEncoding(nn.Module):
    """Injects positional information into the token embeddings."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device

        position = torch.arange(max_len, device=self.device).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2, device=self.device) * (-math.log(10000.0) / d_model)) # [d_model/2]
        pe = torch.zeros(max_len, 1, d_model, device=self.device) # [max_len, 1, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe is now [max_len, 1, d_model]
        self.register_buffer('pe', pe) # register_buffer makes it part of state_dict but not parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        Returns:
            Tensor with positional encoding added, shape [batch_size, seq_len, embedding_dim]
        """
        # x is [b, s, e] -> need pe [s, 1, e] sliced and added
        # self.pe has shape [max_len, 1, e]
        x = x + self.pe[:x.size(1)].transpose(0, 1) # Slice pe to seq_len, transpose to [1, s, e] for broadcasting
        return self.dropout(x)


class Transformer(nn.Module):
    """Standard Transformer Encoder-Decoder architecture.

    Parameters:
        d_model: Dimension of embeddings and hidden states.
        nhead: Number of attention heads.
        num_encoder_layers: Number of encoder layers.
        num_decoder_layers: Number of decoder layers.
        dim_feedforward: Dimension of the feedforward network model.
        dropout: Dropout value.
        device: Device to run the model on.
    """
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            device: str = 'cpu',
        ):
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(
            d_model, dim_feedforward, num_encoder_layers, nhead, dropout, device=self.device
        )
        self.decoder = TransformerDecoder(
            d_model, dim_feedforward, num_decoder_layers, nhead, dropout, device=self.device
        )

    def forward(
            self,
            src: torch.FloatTensor, # [b, src_len, dim] embedded source
            tgt: torch.FloatTensor, # [b, tgt_len, dim] embedded target
            tgt_mask: torch.BoolTensor = None, # [tgt_len, tgt_len] causal mask
            src_key_padding_mask: torch.BoolTensor = None, # [b, src_len] padding mask
            tgt_key_padding_mask: torch.BoolTensor = None, # [b, tgt_len] padding mask
            memory_key_padding_mask: torch.BoolTensor = None # [b, src_len] padding mask (same as src_key_padding_mask)
        ) -> torch.FloatTensor:
        """
        Args:
            src: Embedded source sequence [batch_size, src_seq_len, dim_model].
            tgt: Embedded target sequence [batch_size, tgt_seq_len, dim_model].
            tgt_mask: Causal mask for target self-attention [tgt_seq_len, tgt_seq_len]. True masks position.
            src_key_padding_mask: Mask for source padding [batch_size, src_seq_len]. True masks position.
            tgt_key_padding_mask: Mask for target padding [batch_size, tgt_seq_len]. True masks position.
            memory_key_padding_mask: Mask for memory (encoder output) padding [batch_size, src_seq_len]. True masks position.

        Returns:
            Decoder output [batch_size, tgt_seq_len, dim_model].
        """
        if memory_key_padding_mask is None:
            memory_key_padding_mask = src_key_padding_mask

        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return output


class TranslationTransformer(nn.Module):
    """Transformer wrapper for translation, handling embeddings and masks.

    Parameters:
        n_tokens_src: Size of the source vocabulary.
        n_tokens_tgt: Size of the target vocabulary.
        n_heads: Number of attention heads.
        dim_embedding: Dimension of token embeddings.
        dim_hidden: Dimension of the feedforward network model in Transformer blocks.
        n_layers: Number of encoder and decoder layers.
        dropout: Dropout value.
        src_pad_idx: Index of the padding token in the source vocabulary.
        tgt_pad_idx: Index of the padding token in the target vocabulary.
        device: Device to run the model on.
    """
    def __init__(
            self,
            n_tokens_src: int,
            n_tokens_tgt: int,
            n_heads: int,
            dim_embedding: int,
            dim_hidden: int,
            n_layers: int,
            dropout: float,
            src_pad_idx: int,
            tgt_pad_idx: int,
            device: str = 'cpu',
        ):
        super().__init__()
        self.device = device
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.src_embedding = nn.Embedding(n_tokens_src, dim_embedding, padding_idx=src_pad_idx, device=self.device)
        self.tgt_embedding = nn.Embedding(n_tokens_tgt, dim_embedding, padding_idx=tgt_pad_idx, device=self.device)
        self.position_embedding = PositionalEncoding(dim_embedding, dropout, device=self.device)

        self.transformer = Transformer(
            d_model=dim_embedding,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=dim_hidden,
            dropout=dropout,
            device=self.device
        )

        self.final_out = nn.Linear(dim_embedding, n_tokens_tgt, device=self.device)

        # Initialize weights (optional but often helpful)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
            self,
            source: torch.LongTensor, # [b, src_len]
            target: torch.LongTensor # [b, tgt_len]
        ) -> torch.FloatTensor:
        """
        Args:
            source: Source token indices [batch_size, src_seq_len].
            target: Target token indices [batch_size, tgt_seq_len].

        Returns:
            Output logits [batch_size, tgt_seq_len, n_tokens_tgt].
        """
        src_key_padding_mask = self.generate_key_padding_mask(source, self.src_pad_idx)
        tgt_key_padding_mask = self.generate_key_padding_mask(target, self.tgt_pad_idx)
        tgt_mask = self.generate_causal_mask(target.size(1)) # Causal mask depends only on target length

        # Embed and add positional encoding
        # Scaling embedding as in original paper (optional)
        src_emb = self.position_embedding(self.src_embedding(source) * math.sqrt(self.transformer.encoder.layers[0].self_attn.dim_head * self.transformer.encoder.layers[0].self_attn.n_heads)) # Use d_model
        tgt_emb = self.position_embedding(self.tgt_embedding(target) * math.sqrt(self.transformer.decoder.layers[0].self_attn.dim_head * self.transformer.decoder.layers[0].self_attn.n_heads)) # Use d_model


        # Transformer forward pass
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask # Pass src padding mask for memory
        )

        # Final linear layer
        logits = self.final_out(output)
        return logits

    def generate_causal_mask(self, size: int) -> torch.BoolTensor:
        """Generates an upper-triangular matrix for causal masking."""
        mask = torch.triu(torch.ones(size, size, device=self.device, dtype=torch.bool), diagonal=1)
        return mask # Shape [size, size], True indicates masking

    def generate_key_padding_mask(self, tokens: torch.LongTensor, pad_idx: int) -> torch.BoolTensor:
        """Generates a mask for padding tokens."""
        mask = (tokens == pad_idx)
        return mask # Shape [batch_size, seq_len], True indicates padding
