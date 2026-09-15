import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def forecast_linear_trend(df: pd.DataFrame, date_col: str, value_col: str, periods_ahead: int = 3) -> dict:
    """
    Fits linear regression on time series data to project future values.
    """
    if date_col not in df.columns or value_col not in df.columns:
        return {"error": "Date or Value column not found."}

    temp = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "val": pd.to_numeric(df[value_col], errors="coerce")
    }).dropna().sort_values("date")

    if len(temp) < 4:
        return {"error": "Insufficient data points for trend forecasting (minimum 4 required)."}

    # Resample monthly or ordinal
    temp["ordinal"] = np.arange(len(temp))
    X = temp[["ordinal"]].values
    y = temp["val"].values

    model = LinearRegression()
    model.fit(X, y)

    r_squared = model.score(X, y)
    future_ordinals = np.arange(len(temp), len(temp) + periods_ahead).reshape(-1, 1)
    predictions = model.predict(future_ordinals)

    forecast_results = [
        {"period": i + 1, "projected_value": round(float(pred), 2)}
        for i, pred in enumerate(predictions)
    ]

    trend_direction = "Upward Positive Growth" if model.coef_[0] > 0 else "Downward Decline"

    return {
        "slope": round(float(model.coef_[0]), 4),
        "r_squared": round(float(r_squared), 4),
        "trend_direction": trend_direction,
        "forecast_results": forecast_results,
        "summary": f"Projection indicates an {trend_direction} with {round(r_squared*100, 1)}% model variance fit."
    }

def cluster_segmentation(df: pd.DataFrame, feature_cols: list, n_clusters: int = 3) -> dict:
    """
    Performs K-Means clustering segmentation across numeric features.
    """
    valid_cols = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(valid_cols) < 1:
        return {"error": "No valid numeric features for clustering."}

    sub_df = df[valid_cols].dropna()
    if len(sub_df) < n_clusters:
        return {"error": "Dataset rows fewer than number of clusters requested."}

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(sub_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)

    sub_df_copy = sub_df.copy()
    sub_df_copy["Cluster"] = labels
    cluster_means = sub_df_copy.groupby("Cluster").mean().round(2).to_dict()

    return {
        "features_used": valid_cols,
        "n_clusters": n_clusters,
        "cluster_counts": {f"Cluster_{k}": int(v) for k, v in pd.Series(labels).value_counts().items()},
        "cluster_means": cluster_means,
    }

def detect_anomalies_zscore(df: pd.DataFrame, column: str, threshold: float = 2.5) -> dict:
    """
    Detects anomaly records exceeding Z-Score threshold.
    """
    if column not in df.columns:
        return {"error": f"Column '{column}' not found."}

    s = pd.to_numeric(df[column], errors="coerce")
    mean_val = s.mean()
    std_val = s.std()

    if std_val == 0 or pd.isna(std_val):
        return {"anomaly_count": 0, "anomalies": []}

    z_scores = (s - mean_val) / std_val
    anomalies_mask = z_scores.abs() > threshold
    anomaly_rows = df[anomalies_mask]

    return {
        "column": column,
        "threshold_zscore": threshold,
        "anomaly_count": int(anomalies_mask.sum()),
        "anomaly_percentage": round(float(anomalies_mask.mean() * 100), 2),
        "anomalies_sample": anomaly_rows.head(5).to_dict(orient="records"),
    }
