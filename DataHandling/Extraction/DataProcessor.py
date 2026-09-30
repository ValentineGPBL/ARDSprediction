import csv

import pandas as pd
import ast

class DataPreprocessor:
    def __init__(self):
        pass

    def load_data(self):
        patients   = pd.read_csv("placeholder.csv")
        admissions = pd.read_csv("placeholder.csv")
        code_icd   = pd.read_csv("placeholder.csv")
        diagnose   = pd.read_csv("placeholder.csv")
        cxr        = pd.read_csv("placeholder.csv")
        return cxr, patients, admissions, code_icd, diagnose

    def clean_cxr(self, cxr):
        cxr = cxr.drop(columns=[c for c in cxr.columns if "unnamed" in c.lower()])

        def extract_all_images(val):
            try:
                paths = ast.literal_eval(val)
                return paths if isinstance(paths, list) else None
            except:
                return None

        cxr["image_list"] = cxr["image"].apply(extract_all_images)
        cxr = cxr.dropna(subset=["image_list", "subject_id", "text"])
        return cxr[["subject_id", "image_list", "text"]]

    def extract_study_id(self, image_path):
        try:
            parts = image_path.split("/")
            for part in parts:
                if part.startswith("s") and part[1:].isdigit():
                    return part
            return None
        except:
            return None

    def expand_images(self, cxr):
        rows = []
        for _, row in cxr.iterrows():
            study_groups = {}
            for image_path in row["image_list"]:
                study_id = self.extract_study_id(image_path)
                if study_id not in study_groups:
                    study_groups[study_id] = []
                study_groups[study_id].append(image_path)

            sorted_studies = sorted(study_groups.keys())
            for i, study_id in enumerate(sorted_studies):
                rows.append({
                    "subject_id"  : row["subject_id"],
                    "study_id"    : study_id,
                    "visit_number": i + 1,
                    "image_path"  : study_groups[study_id][0],
                    "text"        : row["text"],
                })

        df = pd.DataFrame(rows)
        study_counts   = df.groupby("subject_id")["study_id"].nunique()
        valid_patients = study_counts[study_counts > 1].index
        df = df[df["subject_id"].isin(valid_patients)]
        print(f"Patients with more than 1 visit: {df['subject_id'].nunique()}")
        return df

    def load_fio2_pao2(self):
        fio2_chunks = []
        for chunk in pd.read_csv(
            "placeholder.csv",
            chunksize=100_000,
            low_memory=False,
            usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        ):
            fio2_chunks.append(chunk[chunk["itemid"] == 223835])
        fio2 = pd.concat(fio2_chunks)
        fio2 = fio2.rename(columns={"valuenum": "fio2"})
        fio2["charttime"] = pd.to_datetime(fio2["charttime"])
        print(f"FiO2 loaded: {len(fio2)} rows")

        pao2_chunks = []
        for chunk in pd.read_csv(
            "placeholder.csv",
            chunksize=100_000,
            low_memory=False,
            usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        ):
            pao2_chunks.append(chunk[chunk["itemid"] == 50821])
        pao2 = pd.concat(pao2_chunks)
        pao2 = pao2.rename(columns={"valuenum": "pao2"})
        pao2["charttime"] = pd.to_datetime(pao2["charttime"])
        print(f"PaO2 loaded: {len(pao2)} rows")

        return fio2, pao2

    def load_vitals(self):
        vital_map = {
            220045: "heart_rate",
            220210: "resp_rate",
            220277: "spo2",
            220052: "map",
            220179: "sbp",
            220180: "dbp",
        }
        chunks = []

        for chunk in pd.read_csv(
                "placeholder.csv",
                chunksize=100_000,
                low_memory=False,
                usecols=[
                    "subject_id",
                    "hadm_id",
                    "itemid",
                    "valuenum",
                ],
        ):
            chunk = chunk[chunk["itemid"].isin(vital_map)]
            chunks.append(chunk)

        vitals = pd.concat(chunks)

        vitals["variable"] = vitals["itemid"].map(vital_map)

        vitals = (
            vitals
            .groupby(
                ["subject_id", "hadm_id", "variable"]
            )["valuenum"]
            .mean()
            .reset_index()
            .pivot(
                index=["subject_id", "hadm_id"],
                columns="variable",
                values="valuenum"
            )
            .reset_index()
        )
        print(f"Vitals loaded: {len(vitals)} admissions")

        return vitals


    def load_labs(self):

        lab_map = {
            51222: "hemoglobin",
            51300: "wbc",
            51265: "platelets",
            50912: "creatinine",
            51006: "bun",
            50983: "sodium",
            50971: "potassium",
            50931: "glucose",
            50862: "albumin",
        }

        chunks = []

        for chunk in pd.read_csv(
                "placeholder.csv",
                chunksize=100_000,
                low_memory=False,
                usecols=[
                    "subject_id",
                    "hadm_id",
                    "itemid",
                    "valuenum",
                ],
        ):
            chunk = chunk[
                chunk["itemid"].isin(lab_map.keys())
            ]

            chunks.append(chunk)

        labs = pd.concat(chunks, ignore_index=True)

        labs["variable"] = labs["itemid"].map(lab_map)

        labs = (
            labs
            .groupby(
                ["subject_id", "hadm_id", "variable"]
            )["valuenum"]
            .mean()
            .reset_index()
            .pivot(
                index=["subject_id", "hadm_id"],
                columns="variable",
                values="valuenum"
            )
            .reset_index()
        )
        print(f"Labs loaded: {len(labs)} admissions")

        return labs

    def merge_data(self, df, patients, admissions, code_icd, diagnose):
        diagnoses = pd.merge(
            code_icd,
            diagnose[["icd_code", "icd_version", "long_title"]],
            on=["icd_code", "icd_version"],
            how="left",
        )

        pneumonia     = diagnoses[diagnoses["long_title"].str.contains("pneumonia", case=False, na=False)]
        pneumonia_ids = set(pneumonia["subject_id"].unique())
        patients["label"] = patients["subject_id"].isin(pneumonia_ids).astype(int)

        primary_diagnosis = (
            diagnoses.sort_values("seq_num")
            .drop_duplicates(subset="subject_id", keep="first")[["subject_id", "long_title"]]
            .rename(columns={"long_title": "primary_diagnosis"})
        )

        admissions["admittime"] = pd.to_datetime(admissions["admittime"])
        admissions_sorted = admissions.sort_values(
            ["subject_id", "admittime"]
        )[["subject_id", "admittime"]].copy()
        admissions_sorted["visit_number"] = (
            admissions_sorted.groupby("subject_id").cumcount() + 1
        )

        df = pd.merge(df, admissions_sorted, on=["subject_id", "visit_number"], how="left")
        df = df.dropna(subset=["admittime"])

        patients_with_all_dates = df.groupby("subject_id")["admittime"].apply(
            lambda x: x.notna().all()
        )
        valid_patients = patients_with_all_dates[patients_with_all_dates].index
        df = df[df["subject_id"].isin(valid_patients)]

        df = pd.merge(df, patients[["subject_id", "gender", "anchor_age", "label"]],
                      on="subject_id", how="inner")
        df = pd.merge(df, primary_diagnosis, on="subject_id", how="left")

        print(f"Final dataset: {len(df)} rows, {df['subject_id'].nunique()} patients")
        return df

    def merge_fio2_pao2(self, df, fio2, pao2):
        #per-visit FiO2/PaO2 and compute PF ratio
        df = self._merge_and_clean(df, fio2, pao2)
        self._print_pf_ratio_sanity_checks(df)
        return df, None, None

    def apply_fio2_pao2_imputation(self, df, fio2, pao2, fio2_median=None, pao2_median=None):
       
        return self._merge_and_clean(df, fio2, pao2)

    def _merge_and_clean(self, df, fio2, pao2):
        admissions = pd.read_csv(
            "placeholder.csv",
            usecols=["subject_id", "hadm_id", "admittime"],
        )
        admissions["admittime"] = pd.to_datetime(admissions["admittime"])

        df = pd.merge(
            df,
            admissions[["subject_id", "hadm_id", "admittime"]],
            on=["subject_id", "admittime"],
            how="left",
        )

        fio2_per_visit = fio2.groupby(["subject_id", "hadm_id"])["fio2"].mean().reset_index()
        pao2_per_visit = pao2.groupby(["subject_id", "hadm_id"])["pao2"].mean().reset_index()

        df = pd.merge(df, fio2_per_visit, on=["subject_id", "hadm_id"], how="left")
        df = pd.merge(df, pao2_per_visit, on=["subject_id", "hadm_id"], how="left")

        vitals = self.load_vitals()
        labs = self.load_labs()

        df = pd.merge(
            df,
            vitals,
            on=["subject_id", "hadm_id"],
            how="left"
        )

        df = pd.merge(
            df,
            labs,
            on=["subject_id", "hadm_id"],
            how="left"
        )

        df.loc[~df["fio2"].between(21, 100), "fio2"] = float("nan")
        df.loc[df["pao2"] <= 0,              "pao2"] = float("nan")

        before = len(df)
        df = df.dropna(subset=["fio2", "pao2"]).copy()
        print(f"Dropped {before - len(df)} rows without valid FiO2/PaO2 "
              f"({len(df)} remain, {df['subject_id'].nunique()} patients)")

        df["fio2_fraction"] = df["fio2"] / 100.0
        df["pf_ratio"]      = df["pao2"] / df["fio2_fraction"]

        return df

    @staticmethod
    def _print_pf_ratio_sanity_checks(df):
     
        print("\n PF ratio extraction sanity checks ")

        #variability: if these are ~1, imputation snuck back in
        print(f"  Unique FiO2 values     : {df['fio2'].nunique()}")
        print(f"  Unique PaO2 values     : {df['pao2'].nunique()}")
        print(f"  Unique pf_ratio values : {df['pf_ratio'].nunique()}")

        #bounds: FiO2 must be percentage (21–100), PaO2 positive
        assert df['fio2'].between(21, 100).all(), "FiO2 outside [21,100] survived filter"
        assert (df['pao2'] > 0).all(),             "Non-positive PaO2 survived filter"
        print(f"  FiO2 in [21,100] ✓  PaO2 > 0 ✓")

        #Formula: pf_ratio == pao2 / (fio2/100)
        recomputed = df['pao2'] / (df['fio2'] / 100.0)
        max_err = (df['pf_ratio'] - recomputed).abs().max()
        assert max_err < 1e-6, f"pf_ratio formula mismatch: max err {max_err}"
        print(f"  pf_ratio = pao2 / (fio2/100) ✓ (max err {max_err:.2e})")

        #Distribution: hould not be dominated by a single value
        q25, q50, q75 = df['pf_ratio'].quantile([0.25, 0.5, 0.75])
        print(f"  pf_ratio Q1/median/Q3  : {q25:.1f} / {q50:.1f} / {q75:.1f}")
        if q25 == q50 == q75:
            print("  ⚠ Q1=median=Q3 — distribution is collapsed; check for hidden imputation.")

        #Berlin ARDS buckets: should span all four bins, not pile in one
        bins = {
            ">300 (no ARDS)":     (df['pf_ratio'] >  300).sum(),
            "200–300 (mild)":     ((df['pf_ratio'] > 200) & (df['pf_ratio'] <= 300)).sum(),
            "100–200 (moderate)": ((df['pf_ratio'] > 100) & (df['pf_ratio'] <= 200)).sum(),
            "≤100 (severe)":      (df['pf_ratio'] <= 100).sum(),
        }
        print("  Berlin ARDS distribution:")
        for k, v in bins.items():
            pct = 100 * v / len(df)
            print(f"    {k:22s}: {v:>6d} ({pct:5.1f}%)")

        #Inter-patient variability: same patient should have varying values
        #    across admissions (otherwise admission-level merge is broken).
        per_patient_uniq = df.groupby('subject_id')['pf_ratio'].nunique()
        multi_visit = per_patient_uniq[per_patient_uniq.index.isin(
            df.groupby('subject_id').size()[df.groupby('subject_id').size() > 1].index
        )]
        if len(multi_visit) > 0:
            varying = (multi_visit > 1).sum()
            print(f"  Multi-visit patients with varying pf_ratio: "
                  f"{varying}/{len(multi_visit)} ({100*varying/len(multi_visit):.1f}%)")

    def save_output(self, df):
    
        required = [
            "subject_id",
            "visit_number",
            "admittime",
            "image_path",
            "text",
            "gender",
            "anchor_age",
            "primary_diagnosis",

            "fio2",
            "pao2",
            "pf_ratio",

            "heart_rate",
            "resp_rate",
            "spo2",
            "sbp",
            "dbp",
            "map",

            "hemoglobin",
            "wbc",
            "platelets",
            "creatinine",
            "bun",
            "sodium",
            "potassium",
            "glucose",
            "albumin",
        ]
        encoded = [
            "gender_encoded", "age_normalized",
            "diagnosis_encoded", "visit_normalized",
        ]

        missing_encoded = [c for c in encoded if c not in df.columns]
        if missing_encoded:
            print(
                f"Warning: encoded columns not yet computed, skipping: {missing_encoded}\n"
                f"  → Call MetadataFeatureVector.fit_transform() before save_output()."
            )

        cols = required + [c for c in encoded if c in df.columns]
        output = df[cols]
        output.to_csv("placeholder.csv", index=False)
        print(f"saved {len(output)} rows, {output['subject_id'].nunique()} patients")
        print(f"PF ratio range: {output['pf_ratio'].min():.1f} – {output['pf_ratio'].max():.1f}")
