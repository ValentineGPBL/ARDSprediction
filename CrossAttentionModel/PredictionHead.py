import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


ARDS_THRESHOLDS = {
    "severe"   : 100.0,   # PF < 100  — severe ARDS
    "moderate" : 200.0,   # 100 ≤ PF < 200 — moderate ARDS
    "mild"     : 300.0,   # 200 ≤ PF < 300 — mild ARDS
    # PF ≥ 300 — no ARDS / normal oxygenation
}
N_ARDS_CLASSES = 4        # severe / moderate / mild / normal


@dataclass
class PFPrediction:

    pf_pred        : torch.Tensor
    log_pf_mean    : torch.Tensor
    log_pf_log_var : torch.Tensor
    pf_std         : torch.Tensor
    ards_logits    : Optional[torch.Tensor] = None



class PFRatioPredictionHead(nn.Module):
    LOG_PF_MEAN = 5.1
    LOG_PF_STD  = 0.7

    def __init__(
        self,
        d_model          : int   = 128,
        hidden_dim       : int   = 64,
        dropout          : float = 0.1,
        log_var_min      : float = -6.0,
        log_var_max      : float =  4.0,
        use_ards_aux     : bool  = True,
        ards_loss_weight : float = 0.4,
    ):
        super().__init__()
        self.d_model          = d_model
        self.log_var_min      = log_var_min
        self.log_var_max      = log_var_max
        self.use_ards_aux     = use_ards_aux
        self.ards_loss_weight = ards_loss_weight

        self.input_norm = nn.LayerNorm(d_model)

        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        trunk_out = hidden_dim // 2

        self.mean_head    = nn.Linear(trunk_out, 1)
        self.log_var_head = nn.Linear(trunk_out, 1)

        self.ards_head = nn.Linear(trunk_out, N_ARDS_CLASSES) if use_ards_aux else None

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


        nn.init.zeros_(self.mean_head.weight)
        nn.init.constant_(self.mean_head.bias, self.LOG_PF_MEAN)


        nn.init.zeros_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)


    def forward(self, h: torch.Tensor) -> PFPrediction:

        h    = self.input_norm(h)
        feat = self.trunk(h)

        log_pf_mean = self.mean_head(feat).squeeze(-1)

        log_pf_log_var = self.log_var_head(feat).squeeze(-1)
        log_pf_log_var = torch.clamp(
            log_pf_log_var, self.log_var_min, self.log_var_max
        )

        pf_pred = torch.exp(log_pf_mean)
        log_sigma = torch.exp(0.5 * log_pf_log_var)
        # exact std of a log-normal: sqrt(exp(sigma^2) - 1) * exp(mu + sigma^2/2)
        pf_std  = torch.sqrt(torch.expm1(log_sigma ** 2)) * pf_pred * torch.exp(0.5 * log_sigma ** 2)

        ards_logits = self.ards_head(feat) if self.ards_head is not None else None

        return PFPrediction(
            pf_pred        = pf_pred,
            log_pf_mean    = log_pf_mean,
            log_pf_log_var = log_pf_log_var,
            pf_std         = pf_std,
            ards_logits    = ards_logits,
        )

    def compute_loss(
        self,
        pred        : PFPrediction,
        pf_targets  : torch.Tensor,
        loss_mask   : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if loss_mask is not None:
            pf_targets = pf_targets[loss_mask]
            log_pf_mean    = pred.log_pf_mean[loss_mask]
            log_pf_log_var = pred.log_pf_log_var[loss_mask]
            ards_logits    = pred.ards_logits[loss_mask] if pred.ards_logits is not None else None
        else:
            log_pf_mean    = pred.log_pf_mean
            log_pf_log_var = pred.log_pf_log_var
            ards_logits    = pred.ards_logits

        log_pf_true = torch.log(pf_targets.clamp(min=1e-3))

        precision  = torch.exp(-log_pf_log_var)
        sq_error   = (log_pf_true - log_pf_mean) ** 2
        reg_loss   = 0.5 * (precision * sq_error + log_pf_log_var).mean()

        components = {"reg_loss": reg_loss.item()}
        total_loss = reg_loss

        if ards_logits is not None and self.ards_loss_weight > 0:
            ards_labels = pf_to_ards_label(pf_targets)
            # upweight severe and normal, hardest and rarest buckets
            class_weights = torch.tensor(
                [3.0, 1.5, 1.5, 3.0],  # severe, moderate, mild, normal
                device=ards_logits.device,
                dtype=ards_logits.dtype,
            )
            ards_loss   = F.cross_entropy(ards_logits, ards_labels, weight=class_weights)
            total_loss  = total_loss + self.ards_loss_weight * ards_loss
            components["ards_loss"] = ards_loss.item()

        components["total_loss"] = total_loss.item()
        return total_loss, components

    @torch.no_grad()
    def predict(
        self,
        h            : torch.Tensor,
        ci_z         : float = 1.96,
    ) -> dict:

        self.eval()
        pred = self(h)

        log_sigma = torch.exp(0.5 * pred.log_pf_log_var)
        ci_low  = torch.exp(pred.log_pf_mean - ci_z * log_sigma)
        ci_high = torch.exp(pred.log_pf_mean + ci_z * log_sigma)

        ards_class = None
        ards_names = None
        if pred.ards_logits is not None:
            ards_class = pred.ards_logits.argmax(dim=-1)
            labels     = ["severe", "moderate", "mild", "normal"]
            ards_names = [labels[i] for i in ards_class.tolist()]

        rule_based_ards = pf_to_ards_label(pred.pf_pred)

        return {
            "pf_mean"         : pred.pf_pred,
            "pf_std"          : pred.pf_std,
            "log_pf_std"      : log_sigma,   # sigma of log(PF): for calibration metrics
            "pf_ci_low"       : ci_low,
            "pf_ci_high"      : ci_high,
            "ards_class_pred" : ards_class,
            "ards_class_rule" : rule_based_ards,
            "ards_names"      : ards_names,
        }


def pf_to_ards_label(pf: torch.Tensor) -> torch.Tensor:

    labels = torch.full_like(pf, 3, dtype=torch.long)   # default: normal
    labels[pf < ARDS_THRESHOLDS["mild"]]     = 2        # mild
    labels[pf < ARDS_THRESHOLDS["moderate"]] = 1        # moderate
    labels[pf < ARDS_THRESHOLDS["severe"]]   = 0        # severe
    return labels


#ARDS Classification Head
class ARDSClassificationHead(nn.Module):
    """
    Primary 4-class ARDS severity classifier:
        0 = severe   (PF < 100)
        1 = moderate (100 ≤ PF < 200)
        2 = mild     (200 ≤ PF < 300)
        3 = normal   (PF ≥ 300)

    Uses class weights to handle imbalance between severity buckets.
    """

    CLASS_NAMES = ["severe", "moderate", "mild", "normal"]

    def __init__(
        self,
        d_model      : int   = 128,
        hidden_dim   : int   = 64,
        dropout      : float = 0.1,
        class_weights: list  = None,   
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(d_model)

        self.trunk = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Linear(hidden_dim // 2, N_ARDS_CLASSES)

        #class weights stored as buffer so they move to GPU automatically
        weights = torch.tensor(class_weights if class_weights else [2.0, 3.0, 3.0, 1.5],
                               dtype=torch.float32)
        self.register_buffer("class_weights", weights)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, 4)."""
        h    = self.input_norm(h)
        feat = self.trunk(h)
        return self.classifier(feat)

    def compute_loss(
        self,
        logits     : torch.Tensor,
        pf_targets : torch.Tensor,
        loss_mask  : Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:

        if loss_mask is not None:
            logits     = logits[loss_mask]
            pf_targets = pf_targets[loss_mask]

        labels = pf_to_ards_label(pf_targets)

        # ordinal loss: treat severity as ordered (severe < moderate < mild < normal)
        # penalises large jumps more than adjacent misclassifications
        loss = F.cross_entropy(logits, labels, weight=self.class_weights, label_smoothing=0.1)
        return loss, {"total_loss": loss.item()}

    @torch.no_grad()
    def predict(self, h: torch.Tensor) -> dict:
        self.eval()
        logits = self(h)
        probs  = torch.softmax(logits, dim=-1)
        preds  = logits.argmax(dim=-1)
        names  = [self.CLASS_NAMES[i] for i in preds.tolist()]
        return {
            "ards_class_pred" : preds,
            "ards_probs"      : probs,
            "ards_name"       : names,
        }

