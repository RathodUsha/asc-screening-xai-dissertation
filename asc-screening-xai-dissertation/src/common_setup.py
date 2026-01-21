#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================
# src/common_setup.py
# Shared utilities for ASC experiments (ARFF)
# ============================================

import re
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

warnings.filterwarnings("ignore")


# ---------------------------
# Load ARFF
# ---------------------------
def load_arff(path) -> pd.DataFrame:
    """
    Load an ARFF file into a pandas DataFrame and decode bytes -> strings.
    """
    raw_data, meta = arff.loadarff(str(path))
    df = pd.DataFrame(raw_data)

    # Decode bytes to strings for object columns
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]
    return df


# ---------------------------
# Prepare AQ + Demographics
# ---------------------------
def prepare_df_AQ_DEMO(df: pd.DataFrame, dataset_name: str):
    """
    Clean autism ARFF dataframe for AQ+demographic experiments (A2–C2).

    Returns:
        X (DataFrame), y (np.ndarray),
        num_cols (list), cat_cols (list), feature_cols (list)
    """
    df = df.copy()

    # 1) Detect AQ columns: A1_Score ... A10_Score (case-insensitive)
    aq_pattern = re.compile(r"^a(\d+)_score$", re.IGNORECASE)
    aq_cols = []
    for c in df.columns:
        m = aq_pattern.match(c)
        if m:
            aq_cols.append((int(m.group(1)), c))

    if not aq_cols:
        raise ValueError(f"[{dataset_name}] No AQ columns found. Columns: {df.columns.tolist()}")

    aq_cols = [name for (_, name) in sorted(aq_cols, key=lambda t: t[0])]
    print(f"\n[{dataset_name}] AQ columns:", aq_cols)

    # 2) Clean AQ -> 0/1 ints
    for c in aq_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # 3) Drop leaky 'result' column if present (sum of AQ)
    if "result" in df.columns:
        print(f"[{dataset_name}] Dropping leaky column 'result'")
        df = df.drop(columns=["result"])

    # 4) Clean categorical demo (replace ? with Unknown)
    cat_demo = [
        "gender", "ethnicity", "jundice", "austim",
        "contry_of_res", "used_app_before", "age_desc", "relation",
    ]
    for c in cat_demo:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace("?", "Unknown")

    # 5) Age numeric (keep as numeric; impute median)
    if "age" not in df.columns:
        raise ValueError(f"[{dataset_name}] No 'age' column found.")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age"] = df["age"].fillna(df["age"].median())

    # 6) Binary columns: jundice / austim -> 0/1
    def bin_yes_no(col):
        s = df[col].astype(str).str.upper().str.strip()
        return s.map({"NO": 0, "YES": 1}).fillna(0).astype(int)

    df["jundice_bin"] = bin_yes_no("jundice") if "jundice" in df.columns else 0
    df["austim_bin"]  = bin_yes_no("austim") if "austim" in df.columns else 0

    # 7) Label column
    label_candidates = [c for c in df.columns if "class" in c.lower() or "asd" in c.lower()]
    if not label_candidates:
        raise ValueError(f"[{dataset_name}] Could not find label column (class/asd).")

    # Prefer 'Class/ASD' if present
    label_candidates = sorted(label_candidates, key=lambda c: 0 if "class/asd" in c.lower() else 1)
    target_col = label_candidates[0]
    print(f"[{dataset_name}] Using label column: {repr(target_col)}")

    y_raw = df[target_col].astype(str).str.strip().str.upper()
    if set(y_raw.unique()) <= {"YES", "NO"}:
        y = y_raw.map({"NO": 0, "YES": 1}).values
    elif set(y_raw.unique()) <= {"0", "1"}:
        y = y_raw.astype(int).values
    else:
        raise ValueError(f"[{dataset_name}] Unknown label values: {y_raw.unique()}")

    # 8) Feature lists
    num_cols = aq_cols + ["age", "jundice_bin", "austim_bin"]
    cat_cols = ["gender", "ethnicity", "contry_of_res", "used_app_before", "age_desc", "relation"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()

    print(f"[{dataset_name}] X shape: {X.shape}")
    print(f"[{dataset_name}] y counts:", np.bincount(y))

    return X, y, num_cols, cat_cols, feature_cols


def build_preprocessor(num_cols, cat_cols):
    """
    Numeric: StandardScaler
    Categorical: OneHotEncoder(handle_unknown='ignore')
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def run_experiment(X: pd.DataFrame, y: np.ndarray, pipe, exp_name: str,
                   n_splits: int = 5, random_state: int = 42):
    """
    Leak-safe StratifiedKFold CV runner.

    Returns:
        folds_df (DataFrame): per-fold metrics
        summary_df (DataFrame): mean metrics
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_rows = []
    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": [], "pr_auc": []}

    for fold_idx, (tr, te) in enumerate(cv.split(X, y), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        pipe.fit(Xtr, ytr)
        y_pred = pipe.predict(Xte)
        y_proba = pipe.predict_proba(Xte)[:, 1]

        acc = accuracy_score(yte, y_pred)
        prec = precision_score(yte, y_pred, zero_division=0)
        rec = recall_score(yte, y_pred, zero_division=0)
        f1  = f1_score(yte, y_pred, zero_division=0)
        roc = roc_auc_score(yte, y_proba)
        pr  = average_precision_score(yte, y_proba)

        metrics["accuracy"].append(acc)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["f1"].append(f1)
        metrics["roc_auc"].append(roc)
        metrics["pr_auc"].append(pr)

        fold_rows.append({
            "fold": fold_idx,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "pr_auc": pr
        })

        print(f"[{exp_name}] Fold {fold_idx}: "
              f"acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, "
              f"f1={f1:.4f}, roc_auc={roc:.4f}, pr_auc={pr:.4f}")

    summary = {k: float(np.mean(v)) for k, v in metrics.items()}
    print(f"\n[{exp_name}] {n_splits}-fold mean metrics:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    folds_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame([summary])
    return folds_df, summary_df


# In[ ]:




