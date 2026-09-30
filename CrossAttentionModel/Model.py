import torch
import torch.nn as nn
from typing import Optional



MODALITIES = ["text", "image", "metadata"]
N_MODALITIES = len(MODALITIES)

class PerVisitFusion(nn.Module):

    def __init__(
        self,
        d_model          : int   = 128,
        n_heads          : int   = 4,
        n_layers         : int   = 2,
        ffn_mult         : int   = 4,
        dropout          : float = 0.1,
        modality_dropout : float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model          = d_model
        self.modality_dropout = modality_dropout

        self.modality_embeddings = nn.Embedding(N_MODALITIES, d_model)

        self.sentinel_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            for _ in range(N_MODALITIES)
        ])

        self.input_projs = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d_model))
            for _ in range(N_MODALITIES)
        ])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)


        self.aggregation_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

        self.output_norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        text_emb     : torch.Tensor,
        image_emb    : torch.Tensor,
        metadata_emb : torch.Tensor,
        missing_mask : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:


        B = text_emb.size(0)
        device = text_emb.device

        if missing_mask is None:
            missing_mask = torch.zeros(B, N_MODALITIES, dtype=torch.bool, device=device)

        embeddings = [text_emb, image_emb, metadata_emb]

        processed = []
        for i, emb in enumerate(embeddings):
            sentinel = self.sentinel_tokens[i].expand(B, 1, self.d_model).squeeze(1)
            is_missing = missing_mask[:, i].unsqueeze(1).float()
            emb = emb * (1 - is_missing) + sentinel * is_missing
            processed.append(emb)

        if self.training and self.modality_dropout > 0:
            for i in range(N_MODALITIES):
                drop_mask = (torch.rand(B, device=device) < self.modality_dropout)
                all_missing = missing_mask.all(dim=1)
                drop_mask = drop_mask & ~all_missing
                sentinel = self.sentinel_tokens[i].expand(B, 1, self.d_model).squeeze(1)
                processed[i] = torch.where(
                    drop_mask.unsqueeze(1), sentinel, processed[i]
                )

        modality_ids = torch.arange(N_MODALITIES, device=device)
        type_embs    = self.modality_embeddings(modality_ids)

        tokens = []
        for i, emb in enumerate(processed):
            emb = self.input_projs[i](emb)
            emb = emb + type_embs[i]
            tokens.append(emb.unsqueeze(1))

        x = torch.cat(tokens, dim=1)

        key_padding_mask = missing_mask

        all_masked = key_padding_mask.all(dim=1, keepdim=True)
        key_padding_mask = key_padding_mask & ~all_masked


        x = self.transformer(x, src_key_padding_mask=key_padding_mask)


        scores  = self.aggregation_attn(x)

        scores  = scores.masked_fill(
            missing_mask.unsqueeze(-1), float("-inf")
        )
        weights = torch.softmax(scores, dim=1)
        fused   = (x * weights).sum(dim=1)


        return self.output_norm(fused)

