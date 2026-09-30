
from torch.utils.data import Dataset


class ARDSDataset(Dataset):
    def __init__(self, patient_indices, masks, targets, delta_cum, delta_gap):
        # targets is (N_patients, T). one PF value per visit slot
        # a patient is valid if at least one visit has a PF ratio
        valid = ~targets.isnan().all(dim=1)
        valid_list = valid.tolist()
        self.patient_indices = [patient_indices[i] for i, v in enumerate(valid_list) if v]
        self.masks     = masks[valid]
        self.targets   = targets[valid]      # (N_valid, T)
        self.delta_cum = delta_cum[valid]
        self.delta_gap = delta_gap[valid]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.patient_indices[idx], self.masks[idx], self.targets[idx], self.delta_cum[idx], self.delta_gap[idx]

