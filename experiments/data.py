from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def repo_root():
    return Path(__file__).resolve().parents[1]


def default_data_dir():
    return repo_root().parent / "data"


def load_filtered_dataset(data_dir=None):
    data_dir = Path(data_dir) if data_dir else default_data_dir()

    filtered_series_matrix = pd.read_csv(data_dir / "filtered_expression_matrix.csv")
    sample_labels = pd.read_csv(data_dir / "GSE120584_sample_labels.csv")

    X = filtered_series_matrix.set_index("ID_REF").transpose()
    label_lookup = sample_labels.set_index("Sample ID")["LABEL"]
    y = label_lookup.loc[filtered_series_matrix.columns[1:]].to_numpy()

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder


def load_feature_rankings(data_dir=None):
    data_dir = Path(data_dir) if data_dir else default_data_dir()

    mrmr_ranking_df = pd.read_csv(data_dir / "mrmr_feature_ranking.csv")
    mcfs_ranking_df = pd.read_csv(data_dir / "mcfs_feature_ranking.csv")

    return {
        "mRMR": mrmr_ranking_df["feature_name"].tolist(),
        "MCFS": mcfs_ranking_df["feature_name"].tolist(),
    }
