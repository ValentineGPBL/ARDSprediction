import os
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from Embeddings.CNN import CXREncoder

D_MODEL   = 128
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_DIR = "placeholder/path"
CSV_PATH  = os.environ.get("CSV_PATH", "placeholder/path.csv")


def pf_to_class(pf):
    if pf < 100: return 0  #severe
    if pf < 200: return 1  #moderate
    if pf < 300: return 2  #mild
    return 3               #normal


def encode_images(df, img_enc):
    #gradients need to flow through the encoder
    df      = df.reset_index(drop=True)
    valid   = [i for i, p in enumerate(df["image_path"]) if pd.notna(p) and os.path.exists(str(p))]
    embs    = torch.zeros(len(df), img_enc.output_dim, device=DEVICE)

    for i in range(0, len(valid), 16):
        chunk = valid[i:i+16]
        try:
            imgs = CXREncoder.load_batch_from_paths(
                [df["image_path"].iloc[j] for j in chunk], device=DEVICE
            )
            out = img_enc(imgs)
            for k, idx in enumerate(chunk):
                embs[idx] = out[k]
        except Exception as e:
            warnings.warn(f"image batch failed: {e}")

    return embs


class ImageClassifier(nn.Module):
    def __init__(self, d_model=128, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, n_classes)
        )
    

    def forward(self, x):
        return self.net(x)


def main():
    print(f"IMAGE_DIR: {IMAGE_DIR}")
    print(f"Example path: {os.path.join(IMAGE_DIR, 'placeholder/path.jpg')}")
    print(f"Exists: {os.path.exists(os.path.join(IMAGE_DIR, 'placeholder/path.jpg'))}")
    print(f"Image-Only Baseline - unfrozen DenseNet\nDevice: {DEVICE}\n")

    df = pd.read_csv(CSV_PATH)
    df["image_path"] = df["image_path"].apply(lambda p: os.path.join(IMAGE_DIR, str(p)))
    df = df.dropna(subset=["pf_ratio", "image_path"])
    df["label"] = df["pf_ratio"].apply(pf_to_class)

    # check how many images exist
    found = df["image_path"].apply(os.path.exists).sum()
    print(f"Images found on disk: {found} / {len(df)}\n")

    # patient-level split
    patients = df["subject_id"].unique()
    train_p, test_p = train_test_split(patients, test_size=0.2, random_state=42)
    train_p, val_p  = train_test_split(train_p,  test_size=0.25, random_state=42)

    train_df = df[df["subject_id"].isin(train_p)].copy().reset_index(drop=True)
    val_df   = df[df["subject_id"].isin(val_p)].copy().reset_index(drop=True)
    test_df  = df[df["subject_id"].isin(test_p)].copy().reset_index(drop=True)

    train_labels = torch.tensor(train_df["label"].values, dtype=torch.long)

    from torch.utils.data import WeightedRandomSampler, TensorDataset, DataLoader

    #compute weight per sample based on class frequency
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    val_labels   = torch.tensor(val_df["label"].values,   dtype=torch.long)
    test_labels  = torch.tensor(test_df["label"].values,  dtype=torch.long)

    # build encoder and unfreeze last DenseBlock
    img_enc = CXREncoder(output_dim=D_MODEL).to(DEVICE)
    img_enc.unfreeze_stage(1)

    classifier = ImageClassifier(D_MODEL).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(classifier.parameters()) + list(img_enc.parameters()),
        lr=1e-4
    )
    ckpt_path = "placeholder/path.pt"
    best_path = "placeholder/path.pt"

    start_epoch = 1
    best_val_loss = float("inf")

    if os.path.exists(ckpt_path):
        print("loading latest checkpoint")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        classifier.load_state_dict(ckpt["classifier"])
        img_enc.load_state_dict(ckpt["img_enc"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
    elif os.path.exists(best_path):
        print("Loading best checkpoint...")
        ckpt = torch.load(best_path, map_location=DEVICE)
        classifier.load_state_dict(ckpt["classifier"])
        img_enc.load_state_dict(ckpt["img_enc"])

    loss_fn = nn.CrossEntropyLoss()
    train_dataset = TensorDataset(torch.arange(len(train_df)), train_labels)
    train_loader  = DataLoader(train_dataset, batch_size=32, sampler=sampler)

    for epoch in range(start_epoch, 11):
        img_enc.train()
        classifier.train()


        for idx_tensor, label_batch in train_loader:
            idx = idx_tensor.tolist()
            batch_df = train_df.iloc[idx].reset_index(drop=True)
            embs = encode_images(batch_df, img_enc)
            pred = classifier(embs)
            loss = loss_fn(pred, label_batch.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        img_enc.eval()
        classifier.eval()
        with torch.no_grad():
            val_embs = encode_images(val_df, img_enc)
            val_pred = classifier(val_embs)
            val_loss = loss_fn(val_pred, val_labels.to(DEVICE))

        # save best model separately
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "classifier": classifier.state_dict(),
                "img_enc": img_enc.state_dict()
            }, best_path)

        torch.save({
            "epoch": epoch,
            "classifier": classifier.state_dict(),
            "img_enc": img_enc.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }, ckpt_path)


        if epoch % 5 == 0:
            print(f"  epoch {epoch}/10  val_loss={val_loss:.4f}")

    # evaluate on test set
    ckpt = torch.load("placeholder/path.pt", map_location=DEVICE)
    classifier.load_state_dict(ckpt["classifier"])
    img_enc.load_state_dict(ckpt["img_enc"])
    img_enc.eval()
    classifier.eval()

    with torch.no_grad():
        test_embs = encode_images(test_df, img_enc)
        test_pred = classifier(test_embs).argmax(dim=1).cpu().numpy()

    y_test  = test_labels.numpy()
    print("\n Image-Only Baseline Results")
    print(classification_report(
        y_test, test_pred,
        target_names=["severe", "moderate", "mild", "normal"],
        zero_division=0
    ))

    midpoints = {0: 75, 1: 150, 2: 250, 3: 375}
    pred_pf   = np.array([midpoints[p] for p in test_pred])
    true_pf   = test_df["pf_ratio"].values
    mae       = np.mean(np.abs(true_pf - pred_pf))
    medae     = np.median(np.abs(true_pf - pred_pf))
    print(f"MAE  : {mae:.1f} mmHg")
    print(f"MedAE: {medae:.1f} mmHg")


if __name__ == "__main__":
    main()
