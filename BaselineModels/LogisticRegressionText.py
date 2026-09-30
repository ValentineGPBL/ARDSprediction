import ast
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

def pf_to_class(pf):
    if pf < 100: return 0  #severe
    if pf < 200: return 1  #moderate
    if pf < 300: return 2  #mild
    return 3               #normal


def join_reports(val):
    try:
        parts = ast.literal_eval(val)
        return " ".join(parts) if isinstance(parts, list) else str(val)
    except:
        return str(val)


df = pd.read_csv("placeholder.csv")
df = df.dropna(subset=["pf_ratio", "text"])
df["text"]  = df["text"].apply(join_reports)
df["label"] = df["pf_ratio"].apply(pf_to_class)

# patient-level split: same as main pipeline
patients = df["subject_id"].unique()
train_p, test_p = train_test_split(patients, test_size=0.2, random_state=42)
train_p, val_p  = train_test_split(train_p,  test_size=0.25, random_state=42)

train_df = df[df["subject_id"].isin(train_p)]
test_df  = df[df["subject_id"].isin(test_p)]

# TF-IDF: converts text to word frequency vectors, no neural network needed
vectorizer = TfidfVectorizer(max_features=500)
X_train    = vectorizer.fit_transform(train_df["text"])
X_test     = vectorizer.transform(test_df["text"])
y_train    = train_df["label"].values
y_test     = test_df["label"].values

model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Text-Only Logistic Regression Baseline — radiology reports only\n")
print(classification_report(y_test, y_pred, target_names=["severe", "moderate", "mild", "normal"], zero_division=0))

#approximate MAE in mmHg
midpoints = {0: 75, 1: 150, 2: 250, 3: 375}
pred_pf   = np.array([midpoints[p] for p in y_pred])
true_pf   = test_df["pf_ratio"].values
print(f"MAE  : {np.mean(np.abs(true_pf - pred_pf)):.1f} mmHg")
print(f"MedAE: {np.median(np.abs(true_pf - pred_pf)):.1f} mmHg")

print(f"  Text only    (logistic)  : MAE = {np.mean(np.abs(true_pf - pred_pf)):.1f} mmHg")