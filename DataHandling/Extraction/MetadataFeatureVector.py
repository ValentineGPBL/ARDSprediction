import hashlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



class MetadataFeatureVector:

    DIAGNOSIS_HASH_BUCKETS = 512

    def __init__(self):
        self.scaler     = MinMaxScaler()
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "MetadataFeatureVector":

        self.scaler.fit(df[["anchor_age", "visit_number"]])
        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        if not self._is_fitted:
            raise RuntimeError(
                "MetadataFeatureVector must be fitted before transforming"
                "call fit(df_train) first."
            )

        df = df.copy()

        df["gender_encoded"] = (df["gender"] == "M").astype(int)

        df["diagnosis_encoded"] = (
            df["primary_diagnosis"]
            .fillna("Unknown")
            .apply(self._hash_diagnosis)
        )

        scaled = self.scaler.transform(df[["anchor_age", "visit_number"]])
        df["age_normalized"]   = scaled[:, 0]
        df["visit_normalized"] = scaled[:, 1]

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def get_vector(self, row) -> np.ndarray:

        return np.array([
            row["gender_encoded"],
            row["age_normalized"],
            row["diagnosis_encoded"],
            row["visit_normalized"],
        ], dtype=np.float32)

    def get_all_vectors(self, df: pd.DataFrame) -> list:
        return df.apply(self.get_vector, axis=1).tolist()

    @classmethod
    def _hash_diagnosis(cls, diagnosis_str: str) -> int:

        #Maps a diagnosis string to a stable integer in [0, DIAGNOSIS_HASH_BUCKETS).

        h = hashlib.md5(diagnosis_str.encode("utf-8")).digest()
        # reserve bucket 0 for padding_idx-map to 1..DIAGNOSIS_HASH_BUCKETS-1
        return (int.from_bytes(h[:4], "little") % (cls.DIAGNOSIS_HASH_BUCKETS - 1)) + 1