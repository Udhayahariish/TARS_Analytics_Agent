import re
import pandas as pd
import numpy as np
from difflib import get_close_matches
from analysis.data_understanding import analyze_dataset_structure, SYNONYM_DICTIONARY
from analysis.stats_engine import compute_column_statistics, compute_correlation_matrix
from analysis.ml_engine import forecast_linear_trend, cluster_segmentation, detect_anomalies_zscore
from analysis.data_cleaner import clean_dataframe

class DataTools:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_dataset_structure(self) -> dict:
        return analyze_dataset_structure(self.df)

    def clean_dataset_auto(self, target_df: pd.DataFrame = None) -> tuple:
        """Performs automated data cleaning: drops duplicates, imputes missing numeric values."""
        df_to_clean = target_df if target_df is not None else self.df
        return clean_dataframe(df_to_clean)

    def find_column(self, term: str) -> str:
        """Finds closest matching column name using exact, synonym, or fuzzy matching."""
        if not term or self.df.empty:
            return None
        
        t = str(term).lower().strip().replace(" ", "_")
        cols = list(self.df.columns)
        cols_map = {str(c).lower().replace(" ", "_"): c for c in cols}

        # Exact match
        if t in cols_map:
            return cols_map[t]

        # Synonym lookup
        for key, syn_list in SYNONYM_DICTIONARY.items():
            if t == key or t in syn_list:
                for c_clean, orig_c in cols_map.items():
                    if any(syn in c_clean for syn in syn_list):
                        return orig_c

        # Partial substring match
        for c_clean, orig_c in cols_map.items():
            if t in c_clean or c_clean in t:
                return orig_c

        # Fuzzy match
        matches = get_close_matches(t, list(cols_map.keys()), n=1, cutoff=0.5)
        if matches:
            return cols_map[matches[0]]

        return None

    def rank_group_by(self, target_df: pd.DataFrame, cat_col: str, num_col: str, top_n: int = 10, ascending: bool = False) -> pd.DataFrame:
        """Groups dataset by categorical column, aggregates numeric column by sum, and ranks."""
        if target_df.empty or cat_col not in target_df.columns or num_col not in target_df.columns:
            return pd.DataFrame()

        clean_df = target_df.dropna(subset=[cat_col]).copy()
        clean_df[num_col] = pd.to_numeric(clean_df[num_col], errors="coerce").fillna(0)
        
        grouped = clean_df.groupby(cat_col)[num_col].agg(["sum", "count", "mean"]).reset_index()
        grouped = grouped.sort_values(by="sum", ascending=ascending).head(top_n)
        grouped.columns = [cat_col, f"Total {num_col}", "Count", f"Average {num_col}"]
        return grouped

    def time_series_trend(self, target_df: pd.DataFrame, date_col: str, num_col: str, freq: str = "M") -> pd.DataFrame:
        """Aggregates numeric metric over time (monthly, yearly, quarterly)."""
        if target_df.empty or date_col not in target_df.columns or num_col not in target_df.columns:
            return pd.DataFrame()

        temp = target_df.copy()
        temp["parsed_date"] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=["parsed_date"])
        temp[num_col] = pd.to_numeric(temp[num_col], errors="coerce").fillna(0)

        if freq == "Y":
            temp["Period"] = temp["parsed_date"].dt.year.astype(str)
        elif freq == "Q":
            temp["Period"] = temp["parsed_date"].dt.to_period("Q").astype(str)
        else:
            temp["Period"] = temp["parsed_date"].dt.to_period("M").astype(str)

        trend_df = temp.groupby("Period")[num_col].sum().reset_index()
        trend_df.columns = ["Period", f"Total {num_col}"]
        return trend_df.sort_values("Period")

    def aggregate_metric(self, target_df: pd.DataFrame, num_col: str, agg_fn: str = "sum") -> float:
        """Computes metric aggregation (sum, mean, max, min, median, count)."""
        if target_df.empty or num_col not in target_df.columns:
            return 0.0
        
        s = pd.to_numeric(target_df[num_col], errors="coerce").dropna()
        if s.empty:
            return 0.0

        if agg_fn in ["mean", "average", "avg"]:
            return float(s.mean())
        elif agg_fn in ["max", "maximum", "highest"]:
            return float(s.max())
        elif agg_fn in ["min", "minimum", "lowest"]:
            return float(s.min())
        elif agg_fn in ["median"]:
            return float(s.median())
        elif agg_fn in ["count"]:
            return float(len(s))
        else:
            return float(s.sum())

    def get_anomalies_report(self, target_df: pd.DataFrame, num_col: str) -> dict:
        """Detects statistical outliers with confidence scores and explanations."""
        if target_df.empty or num_col not in target_df.columns:
            return {"count": 0, "anomalies": [], "explanation": "No numerical metric found."}

        s = pd.to_numeric(target_df[num_col], errors="coerce").dropna()
        if len(s) < 5 or s.std() == 0:
            return {"count": 0, "anomalies": [], "explanation": "Insufficient variance to detect anomalies."}

        mean_val = s.mean()
        std_val = s.std()
        z_scores = (s - mean_val).abs() / std_val

        outlier_mask = z_scores > 2.5
        outlier_rows = target_df[outlier_mask]

        results = []
        for idx, row in outlier_rows.head(5).iterrows():
            val = row[num_col]
            z_val = z_scores.loc[idx]
            conf = min(99, round(float(80 + (z_val - 2.5) * 5), 1))
            results.append({
                "index": int(idx),
                "value": float(val),
                "z_score": round(float(z_val), 2),
                "confidence_score": conf,
                "detail": f"Value {val} deviates by {z_val:.1f} standard deviations from mean {mean_val:.2f}."
            })

        return {
            "count": int(outlier_mask.sum()),
            "column": num_col,
            "mean": round(float(mean_val), 2),
            "std_dev": round(float(std_val), 2),
            "anomalies": results,
            "explanation": f"Detected {outlier_mask.sum()} statistical anomaly points exceeding 2.5 Z-score threshold in '{num_col}'."
        }

    def run_forecasting(self, target_df: pd.DataFrame, date_col: str, value_col: str, periods: int = 3) -> dict:
        return forecast_linear_trend(target_df if target_df is not None else self.df, date_col, value_col, periods)

    def calculate_statistics(self, column: str) -> dict:
        return compute_column_statistics(self.df, column)

    def calculate_correlation(self) -> dict:
        return compute_correlation_matrix(self.df)
