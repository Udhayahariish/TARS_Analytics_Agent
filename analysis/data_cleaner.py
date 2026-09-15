import re
import pandas as pd
import numpy as np

def clean_dataframe(
    df: pd.DataFrame,
    impute_missing: bool = True,
    drop_duplicates: bool = True,
    clean_currency: bool = True,
    standardize_dates: bool = True
) -> tuple:
    """
    Cleans dataset by handling missing values, duplicates, dates, and currency formatting.
    Returns (cleaned_df, cleaning_report_dict).
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    initial_missing = int(df_clean.isna().sum().sum())

    # 1. Remove duplicate rows
    duplicates_removed = 0
    if drop_duplicates:
        duplicates_removed = int(df_clean.duplicated().sum())
        df_clean = df_clean.drop_duplicates()

    # 2. Strip currency symbols & format numeric strings
    currency_cols_fixed = []
    if clean_currency:
        for col in df_clean.columns:
            if df_clean[col].dtype == object:
                sample_str = df_clean[col].dropna().astype(str).head(20).to_string()
                if any(sym in sample_str for sym in ["$", "₹", "€", "£", ","]):
                    cleaned_series = df_clean[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
                    numeric_series = pd.to_numeric(cleaned_series, errors="coerce")
                    if numeric_series.notna().sum() > 0.5 * len(df_clean):
                        df_clean[col] = numeric_series
                        currency_cols_fixed.append(col)

    # 3. Standardize date strings into YYYY-MM-DD
    dates_fixed = []
    if standardize_dates:
        for col in df_clean.columns:
            if "date" in str(col).lower() or "time" in str(col).lower():
                try:
                    converted = pd.to_datetime(df_clean[col], errors="coerce")
                    if converted.notna().sum() > 0.5 * len(df_clean):
                        df_clean[col] = converted.dt.strftime("%Y-%m-%d")
                        dates_fixed.append(col)
                except Exception:
                    pass

    # 4. Handle Missing Values
    imputed_count = 0
    if impute_missing:
        for col in df_clean.columns:
            if df_clean[col].isna().sum() > 0:
                imputed_count += int(df_clean[col].isna().sum())
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                else:
                    mode_s = df_clean[col].mode()
                    fill_val = mode_s.iloc[0] if not mode_s.empty else "Unknown"
                    df_clean[col] = df_clean[col].fillna(fill_val)

    final_rows = len(df_clean)
    final_missing = int(df_clean.isna().sum().sum())

    report = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "initial_missing_cells": initial_missing,
        "final_missing_cells": final_missing,
        "duplicates_removed": duplicates_removed,
        "imputed_missing_cells": imputed_count,
        "currency_columns_formatted": currency_cols_fixed,
        "date_columns_standardized": dates_fixed,
    }

    return df_clean, report
