import numpy as np
import pandas as pd

TARGET = "C6H6(GT)"


def engineer_features(df):
    """Time-based aur cyclical features add kare."""
    df = df.copy()

    df["Hour"] = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Month"] = df["DateTime"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

    # Cyclical encoding
    df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    return df


def get_feature_target(df):
    """X aur y alag kare."""
    drop_cols = ["Date", "Time", "DateTime", TARGET]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[TARGET]
    return X, y