import pandas as pd
import torch
import numpy as np
from CrossAttentionModel.TimeEmbedding import TimeEmbedding


class VisitTimeEncoder:

    def __init__(self, d_model: int = 128, max_days: float = 3650.0, dropout: float = 0.1):
        self.time_embedding = TimeEmbedding(d_model, max_days, dropout)


    def compute_deltas(
            self,
            df: pd.DataFrame,
            subject_col: str = "subject_id",
            time_col: str = "admittime",
            pad_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.sort_values([subject_col, time_col])

        patients = df[subject_col].unique()
        n_patients = len(patients)
        max_visits = df.groupby(subject_col).size().max()

        cum_matrix = torch.full((n_patients, max_visits), pad_value)
        gap_matrix = torch.full((n_patients, max_visits), pad_value)
        pad_mask = torch.ones(n_patients, max_visits, dtype=torch.bool)

        for i, pid in enumerate(patients):
            visits = df[df[subject_col] == pid][time_col].values
            n = len(visits)

            t0 = visits[0]
            cum_days = torch.tensor(
                [(v - t0) / np.timedelta64(1, "D") for v in visits],
                dtype=torch.float32
            )

            gap_days = torch.zeros(n, dtype=torch.float32)
            gap_days[1:] = cum_days[1:] - cum_days[:-1]

            cum_matrix[i, :n] = cum_days
            gap_matrix[i, :n] = gap_days
            pad_mask[i, :n] = False

        return cum_matrix, gap_matrix, pad_mask

    def embed(
            self,
            df: pd.DataFrame,
            subject_col: str = "subject_id",
            time_col: str = "admittime",
            device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:

        delta_cum, delta_gap, pad_mask = self.compute_deltas(df, subject_col, time_col)

        self.time_embedding.eval()
        with torch.no_grad():
            delta_cum = delta_cum.to(device)
            delta_gap = delta_gap.to(device)
            time_emb = self.time_embedding(delta_cum, delta_gap)

        return time_emb, pad_mask.to(device)


def add_time_to_visit_embeddings(
        fused_visits: torch.Tensor,
        time_emb: torch.Tensor,
) -> torch.Tensor:
    assert fused_visits.shape == time_emb.shape, (
        f"Shape mismatch: fused_visits {fused_visits.shape} "
        f"vs time_emb {time_emb.shape}"
    )
    return fused_visits + time_emb

