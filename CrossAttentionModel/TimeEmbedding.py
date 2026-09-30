import torch
import torch.nn as nn
import numpy as np


class SinusoidalTimeEncoding(nn.Module):

    MAX_DAYS_DEFAULT = 3650.0

    def __init__(self, d_model: int, max_days: float = MAX_DAYS_DEFAULT):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sin/cos pairing"
        self.d_model = d_model
        self.max_days = max_days

        half = d_model // 2
        div_term = torch.exp(
            torch.arange(0, half, dtype=torch.float32)
            * (-np.log(10000.0) / half)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, t_days: torch.Tensor) -> torch.Tensor:

        t = t_days / self.max_days

        t_unsqueezed = t.unsqueeze(-1)
        angles = t_unsqueezed * self.div_term

        encoding = torch.zeros(
            *t.shape, self.d_model,
            dtype=t.dtype,
            device=t.device
        )
        encoding[..., 0::2] = torch.sin(angles)
        encoding[..., 1::2] = torch.cos(angles)

        return encoding



class TimeEmbedding(nn.Module):
    def __init__(
            self,
            d_model: int = 128,
            max_days: float = 3650.0,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.sin_cumulative = SinusoidalTimeEncoding(d_model, max_days)
        self.sin_gap = SinusoidalTimeEncoding(d_model, max_days)

        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
            self,
            delta_cumulative: torch.Tensor,
            delta_gap: torch.Tensor,
    ) -> torch.Tensor:
        enc_cum = self.sin_cumulative(delta_cumulative)
        enc_gap = self.sin_gap(delta_gap)

        combined = torch.cat([enc_cum, enc_gap], dim=-1)
        return self.proj(combined)

