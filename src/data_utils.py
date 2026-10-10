import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def clean_data(df):
    df = df.copy().drop_duplicates()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    for col in df.select_dtypes(include="object").columns:
        if col != "customerID":
            mode = df[col].mode()
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df

def data_quality(df):
    return {
        "Rows": len(df),
        "Columns": len(df.columns),
        "Duplicate rows": int(df.duplicated().sum()),
        "Missing cells": int(df.isna().sum().sum())
    }
