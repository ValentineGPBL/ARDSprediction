import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

#berlin ratio
def pf_to_class(pf):
    if pf<100: return 0  #severe
    if pf<200: return 1  #moderate
    if pf<300: return 2  #mild
    return 3               # normal


df = pd.read_csv("placeholder.csv")
df = df.dropna(subset=["pf_ratio", "anchor_age", "gender", "primary_diagnosis"])
df["label"] = df["pf_ratio"].apply(pf_to_class)
df["gender_enc"] = (df["gender"] == "M").astype(int)
df["diagnosis_enc"] = df["primary_diagnosis"].apply(lambda x: hash(str(x)) % 512)

features = ["anchor_age", "gender_enc", "diagnosis_enc"]
# patient-level split. same as main pipeline
patients = df["subject_id"].unique()
train_p, test_p = train_test_split(patients, test_size=0.2, random_state=42)
train_df = df[df["subject_id"].isin(train_p)]
test_df  = df[df["subject_id"].isin(test_p)]

features = ["anchor_age", "gender_enc", "diagnosis_enc"]
scaler   = StandardScaler()
X_train  = scaler.fit_transform(train_df[features])
X_test   = scaler.transform(test_df[features])
y_train  = train_df["label"].values
y_test   = test_df["label"].values

model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Logistic Regression Baseline: age, gender, diagnosis only\n")
print(classification_report(y_test, y_pred, target_names=["severe", "moderate", "mild", "normal"]))

# approximate MAE in mmHg
midpoints = {0: 75, 1: 150, 2: 250, 3: 375}
pred_pf   = np.array([midpoints[p] for p in y_pred])
true_pf   = test_df["pf_ratio"].values
print(f"MAE  : {np.mean(np.abs(true_pf - pred_pf)):.1f} mmHg")
print(f"MedAE: {np.median(np.abs(true_pf - pred_pf)):.1f} mmHg")
