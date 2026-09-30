import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



class AttentionPool2d(nn.Module):
    #Spatial attention pooling over a (B, C, H, W) feature map.

    def __init__(self, in_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 1),
            nn.Tanh(),
            nn.Conv2d(in_dim // 2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.score(x).flatten(2).softmax(-1)   # (B, 1, H*W)
        return (x.flatten(2) * w).sum(-1)          # (B, C)


class CXREncoder(nn.Module):
    #DenseNet-121 chest X-ray encoder producing a (B, output_dim) embedding.

    def __init__(
        self,
        output_dim: int = 128,
        dropout: float = 0.1,
        pretrained: bool = True,
        chexnet_weights: str = None,
    ):
        super().__init__()
        self.output_dim = output_dim

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)

        self.backbone = nn.Sequential(
            densenet.features,
            nn.ReLU(inplace=True),
        )
        self.backbone_out_dim = 1024

        if chexnet_weights is not None:
            self._load_chexnet(chexnet_weights)

        self.attn_pool = AttentionPool2d(self.backbone_out_dim)

        self.proj = nn.Sequential(
            nn.Linear(self.backbone_out_dim, self.backbone_out_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.backbone_out_dim // 2, output_dim),
            nn.LayerNorm(output_dim),
        )

        self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.attn_pool(self.backbone(x)))

    @staticmethod
    def get_transforms(image_size: int = 224, augment: bool = False) -> T.Compose:
        #Preprocessing pipeline. augment=True adds train-time jitter
        base = [
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        if augment:
            aug = [
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.2, contrast=0.2),
            ]
            return T.Compose(aug + base)
        return T.Compose(base)

    @staticmethod
    def load_image(path: str, image_size: int = 224, augment: bool = False) -> torch.Tensor:
       
        transform = CXREncoder.get_transforms(image_size, augment)
        img = Image.open(path).convert("RGB")
        return transform(img).unsqueeze(0)

    @classmethod
    def load_batch_from_paths(
        cls,
        paths: list,
        image_size: int = 224,
        augment: bool = False,
        device: str = "cpu",
    ) -> torch.Tensor:
   
        imgs = [cls.load_image(p, image_size, augment).squeeze(0) for p in paths]
        return torch.stack(imgs).to(device)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_stage(self, stage: int):
    
        if stage == 1:
            self.freeze_backbone()
            for name, p in self.backbone.named_parameters():
                if "denseblock4" in name or "norm5" in name:
                    p.requires_grad = True
        elif stage == 2:
            for p in self.backbone.parameters():
                p.requires_grad = True
        else:
            raise ValueError("stage must be 1 or 2")

    def get_parameter_groups(self, base_lr: float = 1e-3, decay: float = 0.1):
        return [
            {"params": self.proj.parameters(),      "lr": base_lr},
            {"params": self.attn_pool.parameters(), "lr": base_lr * decay},
            {"params": self.backbone.parameters(),  "lr": base_lr * decay ** 2},
        ]

    def _load_chexnet(self, path: str):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        backbone_dict = {
            k.replace("densenet121.features.", ""): v
            for k, v in state_dict.items()
            if "densenet121.features" in k
        }
        # backbone is nn.Sequential(features, ReLU). features is index 0
        self.backbone[0].load_state_dict(backbone_dict, strict=False)


if __name__ == "__main__":
    encoder = CXREncoder(output_dim=128)
    print(encoder)

    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        emb = encoder(dummy)

    print(f"\nInput : {dummy.shape}")
    print(f"Output: {emb.shape}")

