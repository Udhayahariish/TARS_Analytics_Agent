import numpy as np
import pandas as pd
from scipy import stats

def compute_column_statistics(df: pd.DataFrame, column: str) -> dict:
    """
    Computes statistical metrics for a numeric column.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = pd.to_numeric(df[column], errors="coerce").dropna()
    if s.empty:
        return {"error": f"No numeric data available in '{column}'."}

    mean_val = float(s.mean())
    std_val = float(s.std())
    q1 = float(np.percentile(s, 25))
    q3 = float(np.percentile(s, 75))
    iqr = q3 - q1

    # Detect Outliers using IQR
    outliers_iqr = s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)].tolist()

    return {
        "column": column,
        "count": int(s.count()),
        "mean": round(mean_val, 4),
        "median": round(float(s.median()), 4),
        "mode": float(s.mode().iloc[0]) if not s.mode().empty else np.nan,
        "variance": round(float(s.var()), 4),
        "std_dev": round(std_val, 4),
        "min": float(s.min()),
        "max": float(s.max()),
        "quartile_25_q1": round(q1, 4),
        "quartile_50_q2": round(float(s.median()), 4),
        "quartile_75_q3": round(q3, 4),
        "iqr": round(iqr, 4),
        "outlier_count": len(outliers_iqr),
        "outliers_sample": outliers_iqr[:5],
    }

def compute_correlation_matrix(df: pd.DataFrame) -> dict:
    """
    Computes pairwise Pearson correlation matrix for all numeric columns.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or num_df.shape[1] < 2:
        return {"error": "Insufficient numeric columns for correlation matrix."}

    corr_df = num_df.corr().round(4)
    return {
        "columns": list(corr_df.columns),
        "matrix": corr_df.to_dict(),
        "flat_correlations": [
            {
                "col1": c1,
                "col2": c2,
                "correlation": float(corr_df.loc[c1, c2])
            }
            for c1 in corr_df.columns for c2 in corr_df.columns if c1 < c2
        ]
    }

def compute_growth_metrics(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
    """
    Calculates percentage growth, CAGR, and moving averages over time.
    """
    if date_col not in df.columns or value_col not in df.columns:
        return {"error": "Date or Value column missing."}

    temp = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "val": pd.to_numeric(df[value_col], errors="coerce")
    }).dropna().sort_values("date")

    if len(temp) < 2:
        return {"error": "Not enough valid date-value pairs."}

    first_val = temp["val"].iloc[0]
    last_val = temp["val"].iloc[-1]
    total_pct_change = ((last_val - first_val) / abs(first_val)) * 100 if first_val != 0 else 0.0

    years = max(1.0, (temp["date"].iloc[-1] - temp["date"].iloc[0]).days / 365.25)
    cagr = (((last_val / first_val) ** (1.0 / years)) - 1) * 100 if first_val > 0 and last_val > 0 else 0.0

    temp["moving_avg_3"] = temp["val"].rolling(window=3, min_periods=1).mean()

    return {
        "first_val": round(float(first_val), 2),
        "last_val": round(float(last_val), 2),
        "total_pct_change": round(float(total_pct_change), 2),
        "cagr": round(float(cagr), 2),
        "years": round(years, 2),
        "moving_average_latest": round(float(temp["moving_avg_3"].iloc[-1]), 2),
    }
