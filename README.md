# ARDS Severity Prediction

A multimodal deep learning pipeline for predicting Acute Respiratory Distress Syndrome (ARDS) severity from chest X-rays, radiology reports and structured patient metadata, built on the MIMIC-IV clinical database.

---

## Description

This project predicts ARDS severity using the Berlin Definition four-class scale (severe / moderate / mild / normal) based on the PF ratio. Each patient visit is encoded across three modalities — radiology report text (CXR-BERT), chest X-ray images (DenseNet-121), and structured metadata (MLP) — and fused using either a cross-attention transformer or a Graph Attention Network (GNN/GAT).

Two prediction strategies were evaluated:
- **Direct classification**: a four-class head trained end-to-end with weighted cross-entropy
- **Regression-then-threshold**: continuous PF ratio prediction converted to severity classes via Berlin Definition thresholds

The GNN fusion achieved 47.2% accuracy and a macro-AUC of 0.725 on a held-out test set of 1,599 patient visits.

---

## Project Structure

```
ardsprediction2/
├── main.py                         # Entry point — training and prediction
├── Pipeline.py                     # Full training/evaluation pipeline
├── Evaluation.py                   # Metrics, plots, classification report
├── ARDSDataset.py                  # Dataset class
├── CrossAttentionModel/
│   ├── Model.py                    # Main model definition
│   ├── GNNFusion.py                # GNN/GAT fusion module
│   ├── PredictionHead.py           # Classification and regression heads
│   ├── TemporalTransformer.py      # Temporal transformer encoder
│   ├── TimeEmbedding.py            # Sinusoidal time encoding
│   └── VisitTimeEncoder.py         # Visit-level time encoder
├── Embeddings/
│   ├── CNN.py                      # DenseNet-121 image encoder (CXREncoder)
│   ├── TextEmbeddings.py           # CXR-BERT text encoder
│   └── MetaEmbeddings.py           # Metadata MLP encoder
├── DataHandling/
│   ├── Extraction/                 # Data loading and preprocessing
│   └── Tensors/                    # Sequence dataset and batching
├── BaselineModels/
│   ├── BaselineImageOnly.py        # DenseNet-121 image-only baseline
│   ├── LogisticRegressionText.py   # TF-IDF + logistic regression (text)
│   └── LogisticRegressionMetaData.py # Logistic regression (metadata)
```

---

## Requirements

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas, numpy, matplotlib

Training was run on Google Colab with a GPU runtime. The model uses `bf16` mixed precision when available.

---

## Data

The dataset is a cleaned subset of [MIMIC-IV](https://physionet.org/content/mimiciv/), containing **9,948 real patient visits** across 8,070 unique patients. Access requires a PhysioNet credentialed account.

A data quality issue was identified and corrected during development: approximately 80% of PF ratio values in the original dataset were imputed constants rather than true measurements. All models were trained on the corrected data only.

Set the path to your CSV via the environment variable:
```bash
export CSV_PATH=placeholder/path.csv
```

---

## Usage

**Train the model:**
```bash
# Cross-attention fusion (default)
python main.py -fusion-type transformer

# GNN/GAT fusion
python main.py --fusion-type gat
```

**Evaluate saved predictions:**
```python
from Evaluation import evaluate_classification_from_csv
evaluate_classification_from_csv("placeholder.csv")
```

**Run a baseline:**
```bash
python BaselineModels/LogisticRegressionText.py
python BaselineModels/LogisticRegressionMetaData.py
python BaselineModels/BaselineImageOnly.py
```

---

## Results Summary

| Model | Accuracy | Macro F1 |
|---|---|---|
| Metadata-only (logistic regression) | 23% | 0.19 |
| Image-only (DenseNet-121) | 26% | 0.24 |
| Text-only (TF-IDF + logistic) | 41% | 0.37 |
| Multimodal cross-attention | 42.8% | 0.33 |
| Multimodal GNN (GAT) | 47.2% | 0.37 |
