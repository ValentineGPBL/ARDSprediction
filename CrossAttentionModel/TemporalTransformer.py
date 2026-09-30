import torch
import torch.nn as nn
from typing import Optional, Tuple


class CausalTemporalTransformer(nn.Module):

    def __init__(
        self,
        d_model     : int   = 128,
        n_heads     : int   = 4,
        n_layers    : int   = 4,
        ffn_mult    : int   = 4,
        dropout     : float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_model     = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model     = d_model,
            nhead       = n_heads,
            dim_feedforward = d_model * ffn_mult,
            dropout     = dropout,
            activation   = "gelu",
            batch_first = True,
            norm_first  = True
        )


        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers = n_layers,
            norm        = nn.LayerNorm(d_model), )



    def forward(
        self,
        x            : torch.Tensor,
        padding_mask : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        T = x.size(1)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)

        x = self.transformer(
            x,
            mask = causal_mask,
            src_key_padding_mask = padding_mask,
        )

        return x


    def gather_last_visit(
        self,
        x       : torch.Tensor,
        lengths : torch.Tensor,
    ) -> torch.Tensor:
        idx = (lengths - 1).clamp(min=0)
        idx = idx.unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(-1, 1, x.size(-1))
        return x.gather(1, idx).squeeze(1)

    def gather_all_visits(
        self,
        x            : torch.Tensor,
        padding_mask : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if padding_mask is None:
            B, T, _ = x.shape
            padding_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        real_mask = ~padding_mask
        visits    = x[real_mask]
        origins   = real_mask.nonzero(as_tuple=True)[0]
        return visits, origins