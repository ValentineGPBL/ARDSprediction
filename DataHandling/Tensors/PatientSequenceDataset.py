import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PatientSequenceDataset(Dataset):
    def __init__(self, sequences, lengths, targets):
        self.sequences = sequences
        self.lengths = lengths
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.lengths[idx], self.targets[idx]

def collate_fn(batch):
    sequences, lengths, targets = zip(*batch)
    padded = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.float32)
    return padded, lengths, targets

def build_patient_sequence(df, fusion_tensor):
    sequences, lengths, targets = [], [], []

    for subject_id, group in df.groupby("subject_id"):
        indices = group.index.tolist()
        patient_fused = fusion_tensor[indices]
        target = group["pf_ratio"].iloc[-1]

        if pd.isna(target):
            continue

        sequences.append(patient_fused)
        lengths.append(len(indices))
        targets.append(float(target))

    return sequences, lengths, targets

