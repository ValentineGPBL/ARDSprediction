import argparse

from Pipeline import Pipeline, FUSION_TYPES
import os


CKPT = os.environ.get("CKPT_PATH", "placeholder/path.pt")




def main():
    parser = argparse.ArgumentParser(description="Train ARDS P/F prediction pipeline")
    parser.add_argument(
        "--fusion-type",
        choices=FUSION_TYPES,
        default="gat",
        help="Fusion module: transformer, gat",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    args = parser.parse_args()

    pipeline = Pipeline(d_model=128, batch_size=128, epochs=20, lr=5e-5, use_bf16=True, fusion_type=args.fusion_type)
    print(f"using {pipeline.device} (BF16={pipeline.use_bf16})")

    # resume from checkpoint if one exists (e.g. after a session crash)
    if os.path.exists(CKPT):
        pipeline.load_best(CKPT)
        print(f"Resumed from checkpoint: {CKPT}")

    csv_path = "placeholder.csv"
    train_df, val_df, test_df = pipeline.load_and_split(csv_path)

    IMG_ROOT = "placeholder/path"

    for df in [train_df, val_df, test_df]:
        df["image_path"] = df["image_path"].apply(
            lambda x: os.path.join(IMG_ROOT, x)
        )

    print("building sequences...")

    train_indices, train_masks, train_targets, train_cum, train_gap = pipeline.build_sequences(train_df)
    val_indices,   val_masks,   val_targets,   val_cum,   val_gap   = pipeline.build_sequences(val_df)
    test_indices,  test_masks,  test_targets,  test_cum,  test_gap  = pipeline.build_sequences(test_df)

    train_loader = pipeline.make_loader(train_indices, train_masks, train_targets, train_cum, train_gap, shuffle=True, oversample=True)
    val_loader   = pipeline.make_loader(val_indices,   val_masks,   val_targets,   val_cum,   val_gap)
    test_loader  = pipeline.make_loader(test_indices,  test_masks,  test_targets,  test_cum,  test_gap)

    print("starting training...")
    pipeline.train(train_loader, val_loader, train_df, val_df, ckpt_path=CKPT, unfreeze_at=10)

    pipeline.load_best(CKPT)
    print(f"test loss: {pipeline.run_epoch(test_loader, test_df):.4f}")

    print("Generating predictions...")
    preds_df = pipeline.predict(test_loader, test_df)
    out_path = os.environ.get("PREDICTIONS_PATH", "placeholder/path.csv")
    preds_df.to_csv("placeholder.csv", index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
