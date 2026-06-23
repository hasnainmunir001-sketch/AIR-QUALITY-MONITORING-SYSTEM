import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"


def download_data(data_dir="data"):
    """UCI Air Quality dataset download kare agar local file na ho."""
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "AirQualityUCI.csv")

    if os.path.exists(csv_path):
        return csv_path

    try:
        r = requests.get(UCI_URL, timeout=60)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extract("AirQualityUCI.csv", data_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to download dataset: {e}")

    return csv_path


def load_and_clean(csv_path=None):
    """Dataset load kare, clean kare, datetime parse kare."""
    if csv_path is None or not os.path.exists(csv_path):
        csv_path = download_data()

    df = pd.read_csv(csv_path, sep=";", decimal=",")

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # -200 UCI ka missing value code hai
    df.replace(-200, np.nan, inplace=True)

    target = "C6H6(GT)"
    df[target] = pd.to_numeric(df[target], errors="coerce")

    # DateTime column banao
    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S",
        errors="coerce",
    )

    df.dropna(subset=[target, "DateTime"], inplace=True)
    return df