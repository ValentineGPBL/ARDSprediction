import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from typing import Optional


# BiomedVLP-CXR-BERT-specialized: BERT encoder from BioViL (Bannur et al., 2023)
# Pretrained jointly on CXR images+radiology reports from MIMIC-CXR.
BIOVIL_MODEL_ID = "microsoft/BiomedVLP-CXR-BERT-specialized"



class TokenAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings : (B, seq_len, hidden_dim)
            attention_mask   : (B, seq_len) — 1 for real tokens, 0 for padding
        Returns:
            pooled           : (B, hidden_dim)
        """
        scores = self.attn(token_embeddings)                    # (B, seq_len, 1)
        # Mask padding tokens before softmax — they must not contribute
        scores = scores.masked_fill(
            attention_mask.unsqueeze(-1) == 0, float("-inf")
        )
        weights = torch.softmax(scores, dim=1)                  # (B, seq_len, 1)
        return (token_embeddings * weights).sum(dim=1)          # (B, hidden_dim)


class ClinicalTextEncoder(nn.Module):

    BERT_NUM_LAYERS = 12
    BACKBONE_DIM    = 768

    def __init__(
        self,
        output_dim : int  = 128,
        max_length : int  = 256,
        dropout    : float = 0.1,
        pretrained : bool  = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            BIOVIL_MODEL_ID,
            trust_remote_code=True
        )

        from transformers import AutoConfig
        self.bert = AutoModel.from_pretrained(
            BIOVIL_MODEL_ID,
            output_hidden_states=True,
            return_dict=True,
            trust_remote_code=True

        ) if pretrained else AutoModel.from_config(
            AutoConfig.from_pretrained(
                BIOVIL_MODEL_ID,
                trust_remote_code=True,
                return_dict=True
            )
        )

        self.attn_pool = TokenAttentionPool(self.BACKBONE_DIM)

        self.proj = nn.Sequential(
            nn.Linear(self.BACKBONE_DIM, self.BACKBONE_DIM // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.BACKBONE_DIM // 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._finetune_stage = 0
        self.freeze_backbone()

    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
        token_type_ids : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        token_embeddings = outputs.last_hidden_state

        pooled    = self.attn_pool(token_embeddings, attention_mask)
        return self.proj(pooled)

    def tokenize(self, texts: list[str], device: str = "cpu") -> dict:

        return self.tokenizer(
            texts,
            padding      = True,
            truncation   = True,
            max_length   = self.max_length,
            return_tensors = "pt",
        ).to(device)

    @classmethod
    def encode_from_df(
        cls,
        df         : pd.DataFrame,
        text_col   : str = "text",
        batch_size : int = 16,
        device     : str = "cpu",
        **kwargs,
    ) -> torch.Tensor:

        encoder = cls(**kwargs).to(device)
        encoder.eval()
        all_embeddings = []

        texts = df[text_col].tolist()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tokens = encoder.tokenize(batch_texts, device=device)
            with torch.no_grad():
                emb = encoder(**tokens)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    def freeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        self._finetune_stage = 0
        print("ClinicalTextEncoder — Stage 0: backbone frozen.")

    def unfreeze_stage(self, stage: int):

        if stage not in (1, 2, 3):
            raise ValueError("stage must be 1, 2, or 3")

        self.freeze_backbone()  # reset first

        layers_to_unfreeze = {
            1: list(range(10, 12)),    # last 2
            2: list(range(6,  12)),    # last 6
            3: list(range(0,  12)),    # all
        }[stage]

        for layer_idx in layers_to_unfreeze:
            layer = self.bert.bert.encoder.layer[layer_idx]
            for param in layer.parameters():
                param.requires_grad = True

        #unfreeze pooler and embeddings at stage 3
        if stage == 3:
            for param in self.bert.bert.embeddings.parameters():
                param.requires_grad = True

        self._finetune_stage = stage
        print(f"ClinicalTextEncoder — Stage {stage}: "
              f"BERT layers {layers_to_unfreeze} unfrozen.")

    def get_parameter_groups(self, base_lr: float = 1e-3, decay: float = 0.1):
        groups = [{"params": self.proj.parameters(),      "lr": base_lr},
                  {"params": self.attn_pool.parameters(), "lr": base_lr * decay}]

        num_layers = self.BERT_NUM_LAYERS
        for i, layer in enumerate(reversed(self.bert.bert.encoder.layer)):
            lr = base_lr * (decay ** (i + 2))
            groups.append({"params": layer.parameters(), "lr": lr})

        groups.append({
            "params": self.bert.bert.embeddings.parameters(),
            "lr": base_lr * (decay ** (num_layers + 2))
        })
        return groups


