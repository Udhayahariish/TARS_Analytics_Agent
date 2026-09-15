import re
import pandas as pd
import numpy as np

# Common business synonyms map for standard column matching
SYNONYM_DICTIONARY = {
    "sales": ["sales", "revenue", "income", "earnings", "turnover", "money", "amount", "gross_sales"],
    "profit": ["profit", "margin", "net_income", "earnings", "gross_profit", "returns", "bottom_line"],
    "quantity": ["quantity", "units", "count", "volume", "items_sold", "units_sold", "qty"],
    "cost": ["cost", "expense", "expenditure", "cogs", "spending"],
    "price": ["price", "unit_price", "rate", "cost_per_unit", "msrp"],
    "customer": ["customer", "client", "buyer", "user", "patron", "purchaser"],
    "employee": ["employee", "staff", "worker", "rep", "agent", "personnel", "salesperson"],
    "date": ["date", "time", "timestamp", "created_at", "order_date", "day", "month", "year"],
    "region": ["region", "zone", "location", "territory", "area", "city", "country", "state"],
    "product": ["product", "item", "sku", "service", "offering", "category"],
    "salary": ["salary", "pay", "compensation", "wage", "stipend", "remuneration"],
    "rating": ["rating", "score", "feedback", "satisfaction", "nps", "evaluation"],
}

def analyze_dataset_structure(df: pd.DataFrame) -> dict:
    """
    Performs full profiling and metadata generation for uploaded tabular data.
    Detects column types, numeric/date/text, categories, currency, percentage, and synonyms.
    """
    columns_metadata = []
    
    numeric_cols = []
    date_cols = []
    text_cols = []
    category_cols = []
    id_cols = []
    currency_cols = []
    percentage_cols = []

    for col in df.columns:
        series = df[col]
        col_str = str(col).lower()
        non_null_s = series.dropna()
        sample_vals = non_null_s.head(10).astype(str).tolist()

        # Check currency / percentage indicators from string values or column name
        is_currency = any(symbol in "".join(sample_vals) for symbol in ["$", "₹", "€", "£"]) or any(kw in col_str for kw in ["sales", "revenue", "price", "cost", "salary", "amount", "spent"])
        is_percentage = any("%" in v for v in sample_vals) or any(kw in col_str for kw in ["pct", "percent", "percentage", "margin", "rate"])

        # Infer dtype category
        is_numeric = pd.api.types.is_numeric_dtype(series)
        is_datetime = False

        if not is_numeric:
            try:
                converted_dates = pd.to_datetime(non_null_s.head(50), errors="coerce")
                if converted_dates.notna().sum() > len(non_null_s.head(50)) * 0.7:
                    is_datetime = True
            except Exception:
                pass

        if "id" in col_str or col_str.endswith("_code") or col_str.startswith("code_"):
            id_cols.append(col)
            dtype_label = "ID / Key"
        elif is_datetime or "date" in col_str or "time" in col_str:
            date_cols.append(col)
            dtype_label = "Date / Time"
        elif is_numeric:
            if is_currency:
                currency_cols.append(col)
                dtype_label = "Currency / Financial"
            elif is_percentage:
                percentage_cols.append(col)
                dtype_label = "Percentage / Ratio"
            else:
                numeric_cols.append(col)
                dtype_label = "Numeric Metric"
        else:
            if series.nunique() < min(30, max(5, len(df) * 0.2)):
                category_cols.append(col)
                dtype_label = "Categorical Text"
            else:
                text_cols.append(col)
                dtype_label = "Free Text / Descriptor"

        # Generate synonyms and possible user terms
        synonyms = set()
        for base_term, syn_list in SYNONYM_DICTIONARY.items():
            if any(syn in col_str for syn in syn_list):
                synonyms.update(syn_list)
        if not synonyms:
            synonyms = {col_str, col_str.replace("_", " "), col_str.replace("-", " ")}

        columns_metadata.append({
            "name": str(col),
            "data_type": dtype_label,
            "pandas_dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "unique_count": int(series.nunique()),
            "is_currency": is_currency,
            "is_percentage": is_percentage,
            "possible_meanings": list(synonyms),
            "sample_values": sample_vals[:5],
            "business_context": f"Contains {series.nunique()} unique {dtype_label.lower()} entries."
        })

    # Domain Detection Logic
    all_col_strs = " ".join([str(c).lower() for c in df.columns])
    domain = "General Tabular"
    if any(k in all_col_strs for k in ["sales", "revenue", "order", "price", "customer"]):
        domain = "Sales & E-Commerce Analytics"
    elif any(k in all_col_strs for k in ["movie", "film", "imdb", "rating", "cinema"]):
        domain = "Entertainment & Media Analytics"
    elif any(k in all_col_strs for k in ["salary", "employee", "wage", "department", "hr"]):
        domain = "Human Resources & Payroll"
    elif any(k in all_col_strs for k in ["balance", "account", "transaction", "profit", "cogs"]):
        domain = "Financial & Banking"
    elif any(k in all_col_strs for k in ["patient", "doctor", "diagnosis", "hospital"]):
        domain = "Healthcare & Clinical"
    elif any(k in all_col_strs for k in ["inventory", "stock", "warehouse", "sku"]):
        domain = "Supply Chain & Inventory"
    elif any(k in all_col_strs for k in ["campaign", "clicks", "conversion", "impression"]):
        domain = "Marketing & Digital Analytics"

    # Dataset Health Score (0-100)
    total_cells = len(df) * max(1, len(df.columns))
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
    dup_rows = int(df.duplicated().sum())
    dup_pct = (dup_rows / len(df)) * 100 if len(df) > 0 else 0

    score = 100
    score -= min(30, missing_pct * 3)
    score -= min(25, dup_pct * 5)

    # Check numeric outliers
    anomalies_count = 0
    for col in numeric_cols + currency_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) > 5 and s.std() > 0:
            z = (s - s.mean()).abs() / s.std()
            anomalies_count += int((z > 3.0).sum())

    score -= min(20, anomalies_count * 2)
    score = max(10, min(100, int(score)))

    if score >= 90:
        health_category = "Excellent"
    elif score >= 75:
        health_category = "Good"
    elif score >= 60:
        health_category = "Average"
    else:
        health_category = "Poor"

    # Auto Fix Recommendations
    fix_suggestions = []
    if missing_cells > 0:
        fix_suggestions.append(f"Fill or impute {missing_cells:,} missing value entries.")
    if dup_rows > 0:
        fix_suggestions.append(f"Remove {dup_rows:,} duplicate rows from dataset.")
    if anomalies_count > 0:
        fix_suggestions.append(f"Cap or review {anomalies_count} extreme statistical outliers.")
    if not fix_suggestions:
        fix_suggestions.append("Dataset structure is clean and ready for analytics.")

    # Correlation Matrix
    corr_matrix = {}
    all_num = numeric_cols + currency_cols + percentage_cols
    if len(all_num) >= 2:
        try:
            num_df = df[all_num].apply(pd.to_numeric, errors="coerce").dropna()
            if len(num_df) > 3:
                corr_df = num_df.corr().round(2)
                corr_matrix = {
                    "columns": list(corr_df.columns),
                    "values": corr_df.values.tolist()
                }
        except Exception:
            pass

    # Auto KPIs Generation
    kpi_cards = []
    kpi_cards.append({
        "title": "Total Records",
        "value": f"{len(df):,}",
        "subtitle": f"{len(df.columns)} fields",
        "icon": "database"
    })

    if all_num:
        primary_num = all_num[0]
        s_num = pd.to_numeric(df[primary_num], errors="coerce").dropna()
        if not s_num.empty:
            is_curr = primary_num in currency_cols or "sales" in primary_num.lower() or "revenue" in primary_num.lower()
            prefix = "$" if is_curr else ""
            kpi_cards.append({
                "title": f"Total {primary_num}",
                "value": f"{prefix}{s_num.sum():,.2f}" if is_curr else f"{s_num.sum():,.0f}",
                "subtitle": f"Avg: {prefix}{s_num.mean():,.2f}",
                "icon": "trending-up"
            })
            kpi_cards.append({
                "title": f"Average {primary_num}",
                "value": f"{prefix}{s_num.mean():,.2f}",
                "subtitle": f"Max: {prefix}{s_num.max():,.2f}",
                "icon": "bar-chart-2"
            })

    if category_cols:
        cat = category_cols[0]
        top_cat = df[cat].value_counts().head(1)
        if not top_cat.empty:
            kpi_cards.append({
                "title": f"Top {cat}",
                "value": str(top_cat.index[0]),
                "subtitle": f"{int(top_cat.values[0]):,} entries ({round(top_cat.values[0]/len(df)*100, 1)}%)",
                "icon": "award"
            })

    # Compute Summary Stats, Date Bounds, and Categorical Distributions
    stats_summary = {}
    date_summary = {}
    cat_summary = {}

    all_num = numeric_cols + currency_cols + percentage_cols
    for col in all_num:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            stats_summary[col] = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": round(float(s.mean()), 2),
                "sum": round(float(s.sum()), 2)
            }

    for col in date_cols:
        try:
            s_date = pd.to_datetime(df[col], errors="coerce").dropna()
            if not s_date.empty:
                date_summary[col] = {
                    "min_date": str(s_date.min().date()),
                    "max_date": str(s_date.max().date()),
                    "years": sorted(list(set(s_date.dt.year.dropna().astype(int).tolist())))
                }
        except Exception:
            pass

    for col in category_cols:
        top_vals = df[col].value_counts().head(5).to_dict()
        cat_summary[col] = {str(k): int(v) for k, v in top_vals.items()}

    # Executive Summary Markdown
    num_col_desc = f"{len(all_num)} numeric fields ({', '.join(all_num[:4])})" if all_num else "no numeric fields"
    cat_col_desc = f"{len(category_cols)} categorical fields ({', '.join(category_cols[:4])})" if category_cols else "no categorical fields"
    date_col_desc = f"{len(date_cols)} date fields ({', '.join(date_cols[:2])})" if date_cols else "no date fields"

    executive_summary = (
        f"### 🌐 Executive Summary ({domain})\n\n"
        f"I have automatically analyzed **{len(df):,} rows** and **{len(df.columns)} columns**:\n\n"
        f"- **Dataset Health:** **{score}/100** ({health_category} Quality score).\n"
        f"- **Structure:** Contains {num_col_desc}, {cat_col_desc}, and {date_col_desc}.\n"
        f"- **Data Quality:** **{dup_rows} duplicate rows** and **{missing_cells:,} missing values** detected.\n"
    )

    if date_summary:
        first_dcol = list(date_summary.keys())[0]
        dinfo = date_summary[first_dcol]
        executive_summary += f"- **Time Period ({first_dcol}):** Spans from `{dinfo['min_date']}` to `{dinfo['max_date']}`.\n"

    if stats_summary and all_num:
        first_ncol = all_num[0]
        sinfo = stats_summary[first_ncol]
        executive_summary += f"- **Primary Metric ({first_ncol}):** Total = `{sinfo['sum']:,.2f}`, Baseline Avg = `{sinfo['mean']:,.2f}`.\n"


    executive_summary += "\nAsk TARS anything about this dataset or explore the interactive **TARS Analysis** workspace!"

    return {
        "domain": domain,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "health_score": score,
        "health_category": health_category,
        "fix_suggestions": fix_suggestions,
        "duplicate_rows": dup_rows,
        "total_missing": missing_cells,
        "anomalies_count": anomalies_count,
        "numeric_columns": all_num,
        "date_columns": date_cols,
        "categorical_columns": category_cols,
        "text_columns": text_cols,
        "id_columns": id_cols,
        "currency_columns": currency_cols,
        "percentage_columns": percentage_cols,
        "columns_metadata": columns_metadata,
        "stats_summary": stats_summary,
        "date_summary": date_summary,
        "cat_summary": cat_summary,
        "correlation_matrix": corr_matrix,
        "kpi_cards": kpi_cards,
        "executive_summary_markdown": executive_summary
    }


