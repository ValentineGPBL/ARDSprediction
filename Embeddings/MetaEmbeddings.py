import torch
import torch.nn as nn
import pandas as pd


METADATA_FEATURES = [
    "age_normalized",
    "gender_encoded",
    "visit_normalized",
]


# Diagnosis is handled separately via nn.Embedding (see below)
DIAGNOSIS_HASH_BUCKETS = 512   # must match MetadataFeatureVector.DIAGNOSIS_HASH_BUCKETS



class MetadataMLP(nn.Module):

    def __init__(
        self,
        output_dim    : int   = 128,
        hidden_dim    : int   = 64,
        dropout       : float = 0.1,
        diag_emb_dim  : int   = 16,
        feature_cols  : list  = None,
    ):
        super().__init__()

        self.feature_cols  = feature_cols or METADATA_FEATURES
        self.diag_emb_dim  = diag_emb_dim
        numeric_input_dim  = len(self.feature_cols)
        total_input_dim    = numeric_input_dim + diag_emb_dim

        # learned diagnosis embedding. no ordinal relationship
        self.diagnosis_embedding = nn.Embedding(
            num_embeddings = DIAGNOSIS_HASH_BUCKETS,
            embedding_dim  = diag_emb_dim,
            padding_idx    = 0,
        )

        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.residual_proj = nn.Linear(total_input_dim, output_dim)

    def forward(
        self,
        x             : torch.Tensor,
        diagnosis_idx : torch.Tensor,
    ) -> torch.Tensor:

        diag_emb = self.diagnosis_embedding(diagnosis_idx)
        combined = torch.cat([x, diag_emb], dim=-1)
        return self.mlp(combined) + self.residual_proj(combined)

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> tuple["MetadataMLP", torch.Tensor, torch.Tensor]:

        df           = pd.read_csv(path)
        feature_cols = kwargs.pop("feature_cols", METADATA_FEATURES)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in CSV: {missing}")

        x         = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        diag_idx  = torch.tensor(df["diagnosis_encoded"].values, dtype=torch.long)
        model     = cls(feature_cols=feature_cols, **kwargs)
        return model, x, diag_idx