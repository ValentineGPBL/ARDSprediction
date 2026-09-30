import ast
import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from ARDSDataset import ARDSDataset
from DataHandling.Extraction.MetadataFeatureVector import MetadataFeatureVector
from Embeddings.CNN import CXREncoder
from Embeddings.TextEmbeddings import ClinicalTextEncoder
from Embeddings.MetaEmbeddings import MetadataMLP, METADATA_FEATURES
from CrossAttentionModel.Model import PerVisitFusion
from CrossAttentionModel.GNNFusion import build_gnn_fusion
from CrossAttentionModel.TemporalTransformer import CausalTemporalTransformer
from CrossAttentionModel.PredictionHead import ARDSClassificationHead
from CrossAttentionModel.VisitTimeEncoder import VisitTimeEncoder, add_time_to_visit_embeddings

FUSION_TYPES = ("transformer", "gat")




class Pipeline:

    def __init__(
        self,
        d_model=128,
        batch_size=128,
        epochs=10,
        lr=1e-3,
        use_bf16=True,
        fusion_type="gat",
    ):
        if fusion_type not in FUSION_TYPES:
            raise ValueError(f"fusion_type must be one of {FUSION_TYPES}, got {fusion_type!r}")

        self.d_model      = d_model
        self.batch_size   = batch_size
        self.epochs       = epochs
        self.lr           = lr
        self.fusion_type  = fusion_type
        self.device       = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_bf16     = use_bf16 and self.device == "cuda"
        self.text_enc   = ClinicalTextEncoder(output_dim=d_model).to(self.device)
        self.img_enc    = CXREncoder(output_dim=d_model).to(self.device)
        self.meta_mlp   = MetadataMLP(output_dim=d_model).to(self.device)
        self.fusion     = self._build_fusion(fusion_type, d_model).to(self.device)
        self.visit_enc  = VisitTimeEncoder(d_model=d_model)
        self.visit_enc.time_embedding.to(self.device)
        self.transformer= CausalTemporalTransformer(d_model=d_model, n_heads=4, n_layers=4).to(self.device)
        self.head       = ARDSClassificationHead(d_model=d_model, hidden_dim=64).to(self.device)


        self.text_enc.unfreeze_stage(1)
        self.img_enc.unfreeze_stage(1)

        param_groups = []
        param_groups.extend(self.text_enc.get_parameter_groups(base_lr=lr * 0.1, decay=0.5))
        param_groups.extend(self.img_enc.get_parameter_groups(base_lr=lr * 0.1, decay=0.5))

        param_groups.append({"params": self.transformer.parameters(), "lr": lr})
        param_groups.append({"params": self.head.parameters(), "lr": lr})
        param_groups.append({"params": self.meta_mlp.parameters(), "lr": lr})
        param_groups.append({"params": self.fusion.parameters(), "lr": lr})
        param_groups.append({"params": self.visit_enc.time_embedding.parameters(), "lr": lr})

        self.optimizer  = torch.optim.AdamW(param_groups, lr=lr)
        self.scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.all_params = []
        for g in param_groups:
            self.all_params.extend(g["params"])

    @staticmethod
    def _build_fusion(fusion_type, d_model):
        if fusion_type == "transformer":
            return PerVisitFusion(d_model=d_model)

        if fusion_type == "gat":
            return build_gnn_fusion(d_model=d_model)

        raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def run_fusion(
            self,
            text_embs,
            img_embs,
            meta_embs,
            mm,
    ):
        mm = mm.to(self.device)

        return self.fusion(
            text_emb=text_embs,
            image_emb=img_embs,
            metadata_emb=meta_embs,
            missing_mask=mm,
        )

    def load_and_split(self, csv_path):
        df = pd.read_csv(csv_path)

        def join_reports(val):
            try:
                parts = ast.literal_eval(val)
                return " ".join(parts) if isinstance(parts, list) else str(val)
            except:
                return str(val)
        df["text"] = df["text"].apply(join_reports)

        patients = df["subject_id"].unique()
        train_p, test_p = train_test_split(patients, test_size=0.2, random_state=42)
        train_p, val_p  = train_test_split(train_p,  test_size=0.25, random_state=42)

        train_df = df[df["subject_id"].isin(train_p)].copy()
        val_df   = df[df["subject_id"].isin(val_p)].copy()
        test_df  = df[df["subject_id"].isin(test_p)].copy()

        meta     = MetadataFeatureVector()
        train_df = meta.fit_transform(train_df)
        val_df   = meta.transform(val_df)
        test_df  = meta.transform(test_df)

        # leakage check: warn if fio2/pao2 appear in radiology report text
        leak_cols = ["fio2", "pao2", "fio₂", "pao₂", "p/f", "pf ratio"]
        pattern = "|".join(leak_cols)
        n_leaky = df["text"].str.contains(pattern, case=False, na=False).sum()
        if n_leaky > 0:
            import warnings
            warnings.warn(
                f"Potential target leakage: {n_leaky} rows contain FiO2/PaO2 "
                f"references in radiology text. Consider removing these terms."
            )
        else:
            print("Leakage check passed: no FiO2/PaO2 references found in text.")

        return train_df, val_df, test_df

    def _missing_mask(self, df):
        mask = torch.zeros(len(df), 3, dtype=torch.bool)
        mask[:, 0] = torch.tensor(df["text"].isna().values)
        mask[:, 1] = torch.tensor(
            df["image_path"].apply(lambda p: pd.isna(p) or not os.path.exists(str(p))).values
        )
        return mask

    def get_fused_embeddings(self, df, augment=False):
        df = df.reset_index(drop=True)
        mm = self._missing_mask(df)

        # text
        text_embs = []
        for i in range(0, len(df), 64):
            tokens = self.text_enc.tokenize(df["text"].tolist()[i:i+64], device=self.device)
            text_embs.append(self.text_enc(**tokens))
        text_embs = torch.cat(text_embs)

        # images
        img_embs = torch.zeros(len(df), self.img_enc.output_dim, device=self.device)
        valid = [i for i, m in enumerate(mm[:, 1].tolist()) if not m]
        if valid:
            for i in range(0, len(valid), 64):
                chunk = valid[i:i+64]
                try:
                    imgs = CXREncoder.load_batch_from_paths(
                        [df["image_path"].iloc[j] for j in chunk],
                        augment=augment, device=self.device
                    )
                    embs = self.img_enc(imgs)
                    for k, idx in enumerate(chunk):
                        img_embs[idx] = embs[k]
                except Exception as e:
                    warnings.warn(f"image batch failed: {e}")

        # metadata
        x    = torch.tensor(df[METADATA_FEATURES].values, dtype=torch.float32, device=self.device)
        diag = torch.tensor(df["diagnosis_encoded"].values, dtype=torch.long, device=self.device)
        meta_embs = self.meta_mlp(x, diag)

        # fuse
        fused = []
        for i in range(0, len(df), 64):
            sl = slice(i, i+64)
            fused.append(self.fusion(
                text_embs[sl],
                img_embs[sl],
                meta_embs[sl],
                missing_mask=mm[sl].to(self.device)
            ))
        return torch.cat(fused)

    def build_sequences(self, df):
        # enforce sort so visit order matches time delta order from compute_deltas
        df = df.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
        delta_cum, delta_gap, pad_mask = self.visit_enc.compute_deltas(df)

        patients = df["subject_id"].unique()
        T = pad_mask.shape[1]
        # targets is (N_patients, T). nan where visit is padded or PF missing
        targets = torch.full((len(patients), T), float("nan"))
        patient_indices = []

        for b, pid in enumerate(patients):
            idx = df[df["subject_id"] == pid].index.tolist()
            patient_indices.append(idx)
            pf_values = df.loc[idx, "pf_ratio"].values
            for t, pf in enumerate(pf_values):
                if not pd.isna(pf):
                    targets[b, t] = float(pf)

        return patient_indices, pad_mask, targets, delta_cum, delta_gap

    def make_loader(self, patient_indices, masks, targets, delta_cum, delta_gap, shuffle=False, oversample=False,
                    num_workers=2):
        def collate(batch):
            indices = [item[0] for item in batch]
            m = torch.stack([item[1] for item in batch])
            t = torch.stack([item[2] for item in batch])
            c = torch.stack([item[3] for item in batch])
            g = torch.stack([item[4] for item in batch])
            return indices, m, t, c, g

        dataset = ARDSDataset(patient_indices, masks, targets, delta_cum, delta_gap)

        sampler = None
        if oversample:
            # use the last valid PF per patient to determine ARDS class for weighting
            last_pf = []
            for row in dataset.targets:
                valid = row[~row.isnan()]
                last_pf.append(valid[-1] if len(valid) > 0 else torch.tensor(200.0))
            last_pf = torch.stack(last_pf)
            labels = torch.bucketize(
                last_pf,
                boundaries=torch.tensor([100.0, 200.0, 300.0])  # severe/moderate/mild/normal
            )
            class_counts  = torch.bincount(labels, minlength=4).float().clamp(min=1)
            sample_weights = (1.0 / class_counts)[labels]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(dataset),
                replacement=True,
            )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),  # shuffle and sampler are mutually exclusive
            sampler=sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=True,
        )

    def run_epoch(self, loader, df, train=False):
        self.transformer.train() if train else self.transformer.eval()
        self.head.train() if train else self.head.eval()
        self.text_enc.train() if train else self.text_enc.eval()
        self.img_enc.train() if train else self.img_enc.eval()
        self.meta_mlp.train() if train else self.meta_mlp.eval()
        self.fusion.train() if train else self.fusion.eval()
        self.visit_enc.time_embedding.train() if train else self.visit_enc.time_embedding.eval()

        total = 0
        from tqdm import tqdm

        # diagnostic: print image loading rate once per epoch
        if train:
            sample = next(iter(loader))
            indices_sample, masks_sample, _, _, _ = sample
            flat = [idx for indices in indices_sample for idx in indices]
            batch_df = df.iloc[flat].reset_index(drop=True)
            mm = self._missing_mask(batch_df)
            img_missing_rate = mm[:, 1].float().mean().item()
            print(f"  [diag] image missing rate this epoch: {img_missing_rate * 100:.1f}%")

        loader_pbar = tqdm(loader, desc="Training" if train else "Validation", leave=False)
        for indices_list, masks, targets, delta_cum, delta_gap in loader_pbar:
            masks, targets = masks.to(self.device), targets.to(self.device)
            delta_cum, delta_gap = delta_cum.to(self.device), delta_gap.to(self.device)

            flat_indices = [idx for indices in indices_list for idx in indices]
            batch_df = df.iloc[flat_indices].reset_index(drop=True)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                fused_visits = self.get_fused_embeddings(batch_df, augment=train)

                B = len(indices_list)
                T = masks.shape[1]
                seqs = torch.zeros(B, T, self.d_model, device=self.device, dtype=fused_visits.dtype)
                curr = 0
                for b, indices in enumerate(indices_list):
                    n = len(indices)
                    seqs[b, :n] = fused_visits[curr:curr+n]
                    curr += n

                time_emb = self.visit_enc.time_embedding(delta_cum, delta_gap)
                seqs = add_time_to_visit_embeddings(seqs, time_emb)

                out = self.transformer(seqs, padding_mask=masks)  # (B, T, d_model)

                # per-visit supervision: classify ARDS severity at every visit
                # targets is (B, T) — nan where padded or PF missing
                logits       = self.head(out.view(B * T, self.d_model))  # (B*T, 4)
                flat_targets = targets.view(B * T)                        # (B*T,)
                loss_mask    = ~flat_targets.isnan()                      # only real visits
                loss, info   = self.head.compute_loss(logits, flat_targets, loss_mask=loss_mask)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
                self.optimizer.step()

            total += info["total_loss"]

        return total / len(loader)

    def _unfreeze_stage2(self):
        """Unfreeze stage 2 of both encoders and add newly trainable params to optimizer."""
        self.text_enc.unfreeze_stage(2)
        self.img_enc.unfreeze_stage(2)
        new_params = [p for p in list(self.text_enc.parameters()) + list(self.img_enc.parameters())
                      if p.requires_grad and not any(p is q for group in self.optimizer.param_groups for q in group["params"])]
        if new_params:
            self.optimizer.add_param_group({"params": new_params, "lr": self.lr * 0.01})
            self.all_params.extend(new_params)
            print(f"  unfrozen stage 2 — added {len(new_params)} new param tensors to optimizer")

    def train(self, train_loader, val_loader, train_df, val_df, ckpt_path="placeholder.pt", unfreeze_at=None):
        best_val = float("inf")
        for epoch in range(1, self.epochs + 1):

            if unfreeze_at is not None and epoch == unfreeze_at:
                print(f"  epoch {epoch}: unfreezing stage 2 of BERT and CNN backbones")
                self._unfreeze_stage2()

            train_loss = self.run_epoch(train_loader, train_df, train=True)
            val_loss = self.run_epoch(val_loader, val_df)
            self.scheduler.step()
            print(f"epoch {epoch}/{self.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "fusion_type": self.fusion_type,
                    "transformer": self.transformer.state_dict(),
                    "head": self.head.state_dict(),
                    "text_enc": self.text_enc.state_dict(),
                    "img_enc": self.img_enc.state_dict(),
                    "meta_mlp": self.meta_mlp.state_dict(),
                    "fusion": self.fusion.state_dict(),
                    "time_embedding": self.visit_enc.time_embedding.state_dict()
                }, ckpt_path)
                print(f"  saved best model → {ckpt_path}")

    def load_best(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        ckpt_fusion = ckpt.get("fusion_type", "transformer")
        if ckpt_fusion != self.fusion_type:
            warnings.warn(
                f"checkpoint fusion_type={ckpt_fusion!r} differs from pipeline "
                f"fusion_type={self.fusion_type!r}; loading weights anyway"
            )
    def load_best(self, ckpt_path="placeholder.pt"):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.transformer.load_state_dict(ckpt["transformer"])
        self.head.load_state_dict(ckpt["head"])
        self.text_enc.load_state_dict(ckpt["text_enc"])
        self.img_enc.load_state_dict(ckpt["img_enc"])
        self.meta_mlp.load_state_dict(ckpt["meta_mlp"])
        self.fusion.load_state_dict(ckpt["fusion"])
        self.visit_enc.time_embedding.load_state_dict(ckpt["time_embedding"])


    @torch.no_grad()
    def predict(self, loader, df):
        self.transformer.eval()
        self.head.eval()
        self.text_enc.eval()
        self.img_enc.eval()
        self.meta_mlp.eval()
        self.fusion.eval()
        self.visit_enc.time_embedding.eval()

        results = []
        for indices_list, masks, targets, delta_cum, delta_gap in loader:
            masks = masks.to(self.device)
            delta_cum, delta_gap = delta_cum.to(self.device), delta_gap.to(self.device)

            flat_indices = [idx for indices in indices_list for idx in indices]
            batch_df = df.iloc[flat_indices].reset_index(drop=True)

            fused_visits = self.get_fused_embeddings(batch_df, augment=False)

            B = len(indices_list)
            T = masks.shape[1]
            seqs = torch.zeros(B, T, self.d_model, device=self.device)
            curr = 0
            for b, indices in enumerate(indices_list):
                n = len(indices)
                seqs[b, :n] = fused_visits[curr:curr+n]
                curr += n

            time_emb = self.visit_enc.time_embedding(delta_cum, delta_gap)
            seqs = add_time_to_visit_embeddings(seqs, time_emb)

            out = self.transformer(seqs, padding_mask=masks)
            lengths = (~masks).sum(dim=1)
            h = self.transformer.gather_last_visit(out, lengths)
            preds = self.head.predict(h)

            for i in range(B):
                # targets is (B, T) — get last valid PF for this patient
                patient_targets = targets[i]
                valid_pf = patient_targets[~patient_targets.isnan()]
                true_pf  = valid_pf[-1].item() if len(valid_pf) > 0 else float('nan')

                def pf_to_class(pf):
                    if pf < 100: return 0
                    if pf < 200: return 1
                    if pf < 300: return 2
                    return 3

                true_class = pf_to_class(true_pf) if not pd.isna(true_pf) else None
                probs = preds["ards_probs"][i].tolist()
                row = {
                    "true_pf"        : true_pf,
                    "true_class"     : true_class,
                    "pred_class"     : preds["ards_class_pred"][i].item(),
                    "pred_name"      : preds["ards_name"][i],
                    "prob_severe"    : probs[0],
                    "prob_moderate"  : probs[1],
                    "prob_mild"      : probs[2],
                    "prob_normal"    : probs[3],
                }
                results.append(row)

        return pd.DataFrame(results)
