import os
import re
import json
import urllib.request
import pandas as pd
import numpy as np
from agent.tools import DataTools
from analysis.data_understanding import analyze_dataset_structure

# Try importing SDKs
try:
    import google.generativeai as genai
    HAS_GEMINI_SDK = True
except ImportError:
    HAS_GEMINI_SDK = False

try:
    import openai
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

try:
    import ollama
    HAS_OLLAMA_SDK = True
except ImportError:
    HAS_OLLAMA_SDK = False


class DataAnalystAgent:
    def __init__(self, df: pd.DataFrame, gemini_key: str = None, openai_key: str = None):
        self.df = df.copy()
        self.tools = DataTools(self.df)
        self.dataset_meta = self.tools.get_dataset_structure()
        
        # API Keys & Providers
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")

        if self.gemini_key and HAS_GEMINI_SDK:
            try:
                genai.configure(api_key=self.gemini_key)
            except Exception:
                pass

        # State & Memory Management
        self.active_df = self.df.copy()
        self.applied_filters = []
        self.chat_history = []

    def reset_memory(self):
        """Resets active filter states and chat history back to original dataset."""
        self.active_df = self.df.copy()
        self.applied_filters = []
        self.chat_history = []

    def get_dataset_analysis(self) -> dict:
        """Returns full automatic dataset profile understanding."""
        return self.dataset_meta

    def get_power_bi_dashboard(self) -> dict:
        """Generates dynamic Power BI style analytics dashboard payload."""
        df = self.active_df
        meta = self.dataset_meta
        num_cols = meta.get("numeric_columns", [])
        cat_cols = meta.get("categorical_columns", [])
        date_cols = meta.get("date_columns", [])

        primary_num = num_cols[0] if num_cols else (df.columns[0] if not df.empty else "Metric")
        primary_cat = cat_cols[0] if cat_cols else (df.columns[0] if not df.empty else "Category")
        primary_date = date_cols[0] if date_cols else None

        # 1. Trend Line Chart
        trend_chart = None
        if primary_date and primary_num and primary_num in df.columns and primary_date in df.columns:
            tdf = self.tools.time_series_trend(df, primary_date, primary_num, freq="M")
            if not tdf.empty:
                trend_chart = {
                    "type": "line",
                    "title": f"Monthly Trend of {primary_num}",
                    "labels": tdf["Period"].tolist()[:12],
                    "values": tdf[f"Total {primary_num}"].tolist()[:12]
                }

        # 2. Category Bar Chart
        cat_chart = None
        if primary_cat and primary_num and primary_cat in df.columns and primary_num in df.columns:
            rdf = self.tools.rank_group_by(df, primary_cat, primary_num, top_n=8)
            if not rdf.empty:
                cat_chart = {
                    "type": "bar",
                    "title": f"Top 8 {primary_cat} by {primary_num}",
                    "labels": rdf[primary_cat].astype(str).tolist(),
                    "values": rdf[f"Total {primary_num}"].tolist()
                }

        # 3. Distribution Donut Chart
        donut_chart = None
        if primary_cat and primary_cat in df.columns:
            counts = df[primary_cat].value_counts().head(6).to_dict()
            donut_chart = {
                "type": "doughnut",
                "title": f"{primary_cat} Share Distribution",
                "labels": [str(k) for k in counts.keys()],
                "values": list(counts.values())
            }

        # 4. Forecast Chart
        forecast_chart = None
        forecast_info = None
        if primary_date and primary_num and primary_date in df.columns and primary_num in df.columns:
            fc = self.tools.run_forecasting(df, primary_date, primary_num, periods=3)
            if "error" not in fc:
                forecast_info = fc
                results = fc.get("forecast_results", [])
                forecast_chart = {
                    "type": "line",
                    "title": f"3-Period Projected Forecast for {primary_num}",
                    "labels": [f"Period +{r['period']}" for r in results],
                    "values": [r["projected_value"] for r in results]
                }

        # 5. Anomalies
        anomalies_info = self.tools.get_anomalies_report(df, primary_num) if primary_num in df.columns else None

        # 6. Insights & Recommendations
        top_cat_item = cat_chart["labels"][0] if cat_chart and cat_chart["labels"] else "Primary Segment"
        top_cat_val = cat_chart["values"][0] if cat_chart and cat_chart["values"] else 0.0

        ai_insights = [
            f"**{top_cat_item}** is the leading segment with **{top_cat_val:,.2f}** in {primary_num}.",
            f"Dataset quality stands at **{meta['health_score']}/100** ({meta['health_category']} status).",
            f"Indexed **{len(df):,} records** across **{len(df.columns)} fields**."
        ]

        recommendations = [
            f"Focus strategy on **{top_cat_item}** to maximize yield.",
            "Run automated dataset cleaning to resolve any missing value entries.",
            "Review statistical outliers to ensure demand forecast precision."
        ]

        return {
            "domain": meta["domain"],
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "health_score": meta["health_score"],
            "health_category": meta["health_category"],
            "fix_suggestions": meta["fix_suggestions"],
            "kpi_cards": meta["kpi_cards"],
            "trend_chart": trend_chart,
            "category_chart": cat_chart,
            "distribution_chart": donut_chart,
            "forecast_chart": forecast_chart,
            "forecast_info": forecast_info,
            "correlation_matrix": meta["correlation_matrix"],
            "anomalies": anomalies_info,
            "insights": ai_insights,
            "recommendations": recommendations,
            "fields": list(df.columns)
        }

    def ask(self, question: str) -> dict:
        """
        Processes natural language question using Generative AI (Gemini / ChatGPT / Ollama)
        and Pandas execution. Returns ONLY direct, natural, relevant answers without forced section headers.
        """
        q = question.strip()
        q_lower = q.lower()

        # Handle Reset / Start Over
        if any(kw in q_lower for kw in ["reset", "start over", "clear filter", "show all data"]):
            self.reset_memory()
            return {
                "direct_answer": f"Conversational context and filters reset. Now viewing all **{len(self.df):,} rows**.",
                "followups": ["Show top categories", "What is total sales?", "List column names"],
                "active_rows": len(self.df)
            }

        # Step 1: Context & Follow-up Resolution
        resolved_question, is_followup = self._resolve_conversational_context(q)
        
        # If standalone question (not an explicit drill-down follow-up), reset active filter state
        if not is_followup:
            self.active_df = self.df.copy()
            self.applied_filters = []

        # Step 2: Dynamic Filters
        self._apply_dynamic_filters(resolved_question)

        # Step 3: Pandas Analytical Calculations & Lookup
        pandas_result = self._perform_pandas_analysis(resolved_question)

        # Step 4: Generative AI Model Response Synthesis (Gemini / ChatGPT / Ollama / Smart Pandas Fallback)
        ai_response = self._call_generative_model(q, resolved_question, pandas_result)

        # Memory Update
        self.chat_history.append({
            "user_question": q,
            "resolved_question": resolved_question,
            "result_summary": ai_response["direct_answer"]
        })

        return ai_response

    def _resolve_conversational_context(self, question: str) -> tuple:
        if not self.chat_history:
            return question, False

        q_lower = question.lower()
        last_turn = self.chat_history[-1]

        # Explicit drill-down indicators
        is_followup_start = any(q_lower.startswith(w) for w in ["only ", "filter ", "where ", "among ", "just for ", "further ", "how about ", "what about "])
        has_pronoun = any(re.search(r'\b' + p + r'\b', q_lower) for p in ["its", "their", "that", "those", "them", "these", "this"])

        if is_followup_start or has_pronoun:
            return f"{question} (Context: previous query was '{last_turn['user_question']}')", True

        return question, False

    def _apply_dynamic_filters(self, question: str):
        q_lower = question.lower()
        df = self.active_df

        # Date / Year Filter
        year_match = re.search(r'\b(20\d\d|19\d\d)\b', q_lower)
        if year_match:
            year_val = int(year_match.group(1))
            date_cols = self.dataset_meta.get("date_columns", [])
            for dcol in date_cols:
                if dcol in df.columns:
                    try:
                        dt_s = pd.to_datetime(df[dcol], errors="coerce")
                        filtered = df[dt_s.dt.year == year_val]
                        if not filtered.empty:
                            self.active_df = filtered
                            self.applied_filters.append(f"{dcol} Year = {year_val}")
                            break
                    except Exception:
                        pass

        # Numeric Range Filter
        num_match = re.search(r'(under|below|less than|<|over|above|greater than|>)\s*(\d+(?:\.\d+)?)', q_lower)
        if num_match:
            op = num_match.group(1).lower()
            val = float(num_match.group(2))
            is_less = op in ["under", "below", "less than", "<"]

            target_num = self._find_best_numeric_column(q_lower)
            if target_num and target_num in df.columns:
                s = pd.to_numeric(df[target_num], errors="coerce")
                filtered = df[s < val] if is_less else df[s > val]
                if not filtered.empty:
                    self.active_df = filtered
                    self.applied_filters.append(f"{target_num} {'<' if is_less else '>'} {val:,.2f}")

    def _find_best_numeric_column(self, q_lower: str) -> str:
        all_cols = [str(c) for c in self.df.columns]
        num_cols = self.dataset_meta.get("numeric_columns", []) + self.dataset_meta.get("currency_columns", []) + self.dataset_meta.get("percentage_columns", [])
        
        # Ensure list is unique and clean
        seen = set()
        clean_num_cols = []
        for c in num_cols:
            if c in all_cols and c not in seen:
                seen.add(c)
                clean_num_cols.append(c)

        if not clean_num_cols:
            clean_num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(self.df[c])]

        # 1. Exact Column Name Match
        for ncol in clean_num_cols:
            if ncol.lower().strip() in q_lower:
                return ncol

        # 2. Semantic Business Synonym Mapping
        synonym_map = {
            "sales": ["sales", "revenue", "price", "amount", "cost", "total", "value", "money", "earnings", "income", "turnover", "spent", "spending"],
            "profit": ["profit", "margin", "net_income", "earnings", "gross_profit", "returns"],
            "quantity": ["quantity", "units", "count", "volume", "items", "qty", "sold"],
            "price": ["price", "unit_price", "rate", "cost", "msrp", "amount", "val"],
            "engine": ["engine", "power", "horsepower", "hp", "engine_power"],
            "age": ["age", "days", "age_in_days", "old"],
            "distance": ["km", "mileage", "distance", "odometer"],
            "owners": ["owner", "owners", "previous_owners"],
            "rating": ["rating", "score", "feedback", "stars"],
        }

        for concept, syns in synonym_map.items():
            if any(syn in q_lower for syn in syns):
                # Search for matching column among dataset numeric columns
                for col in clean_num_cols:
                    col_l = col.lower().strip()
                    if any(syn in col_l for syn in syns):
                        return col

        # 3. Default Prioritized Fallback (Financial > Volume > General Metric)
        # Never default to coordinates, IDs, zip codes, or dates!
        ignored_kw = ["lat", "lon", "latitude", "longitude", "id", "zip", "postal", "year", "code", "index"]
        valid_candidates = [c for c in clean_num_cols if not any(ikw in c.lower().strip() for ikw in ignored_kw)]

        if not valid_candidates:
            valid_candidates = clean_num_cols

        # Pick financial/currency column if available
        for c in valid_candidates:
            cl = c.lower()
            if any(kw in cl for kw in ["price", "revenue", "sales", "amount", "cost", "salary", "value", "total"]):
                return c

        return valid_candidates[0] if valid_candidates else (all_cols[0] if all_cols else None)

    def _find_best_categorical_column(self, q_lower: str) -> str:
        all_cols = [str(c) for c in self.df.columns]
        cat_cols = self.dataset_meta.get("categorical_columns", []) + self.dataset_meta.get("text_columns", [])
        
        # Clean unique list
        seen = set()
        clean_cat = []
        for c in cat_cols:
            if c in all_cols and c not in seen:
                seen.add(c)
                clean_cat.append(c)

        if not clean_cat:
            clean_cat = [c for c in all_cols if not pd.api.types.is_numeric_dtype(self.df[c])]

        # 1. Direct Column Name Match
        for ccol in clean_cat:
            if ccol.lower().strip() in q_lower:
                return ccol

        # 2. Category Word Synonyms
        cat_synonyms = {
            "model": ["model", "car", "vehicle", "auto", "type", "make", "brand", "product", "item"],
            "category": ["category", "dept", "department", "group", "segment", "class"],
            "region": ["region", "country", "city", "state", "zone", "location"],
            "customer": ["customer", "client", "buyer", "user", "name"]
        }

        for concept, syns in cat_synonyms.items():
            if any(syn in q_lower for syn in syns):
                for col in clean_cat:
                    col_l = col.lower().strip()
                    if any(syn in col_l for syn in syns):
                        return col

        return clean_cat[0] if clean_cat else (all_cols[0] if all_cols else None)

    def _perform_pandas_analysis(self, question: str) -> dict:
        """Executes exact Pandas calculations based on intent."""
        df = self.active_df
        q_lower = question.lower()

        # 1. ENTITY LOOKUP / LIST INTENT (e.g., "car lists", "list cars", "show models", "list categories", "column names")
        is_list_query = any(kw in q_lower for kw in ["list", "lists", "show all", "names", "models", "categories", "items", "cars"])
        target_cat = self._find_best_categorical_column(q_lower)

        if is_list_query and target_cat and target_cat in df.columns:
            counts = df[target_cat].value_counts().head(15).to_dict()
            return {
                "intent": "entity_list",
                "category_column": target_cat,
                "items_with_counts": counts,
                "total_unique": df[target_cat].nunique(),
                "sample_rows": df[[target_cat]].head(10).to_dict(orient="records")
            }

        if any(kw in q_lower for kw in ["how many column", "column names", "list columns", "fields"]):
            return {
                "intent": "column_list",
                "fields": list(df.columns),
                "total_columns": len(df.columns),
                "total_rows": len(df)
            }

        target_num = self._find_best_numeric_column(q_lower)
        target_cat = self._find_best_categorical_column(q_lower)
        target_date = self.dataset_meta["date_columns"][0] if self.dataset_meta["date_columns"] else None

        # 1. SPECIFIC RECORD RETRIEVAL / TOP N RECORD LIST INTENT (e.g. "top 10 car in lounge", "top 5 cars", "show 10 records")
        top_n_match = re.search(r'\b(top|first|show|best|cheapest)\s*(\d+)\b|\b(\d+)\s*(cars?|records?|items?|models?)\b', q_lower)
        n_val = 10
        if top_n_match:
            n_str = top_n_match.group(2) or top_n_match.group(3)
            if n_str:
                n_val = min(50, max(1, int(n_str)))

        # Check if query targets specific categorical value (e.g. "lounge", "pop", "sport")
        matched_filter_col = None
        matched_filter_val = None
        
        cat_cols = self.dataset_meta.get("categorical_columns", []) + self.dataset_meta.get("text_columns", [])
        for ccol in cat_cols:
            if ccol in df.columns:
                unique_vals = df[ccol].dropna().astype(str).unique()
                for uval in unique_vals:
                    if len(uval) >= 2 and re.search(r'\b' + re.escape(uval.lower()) + r'\b', q_lower):
                        matched_filter_col = ccol
                        matched_filter_val = uval
                        break
            if matched_filter_col:
                break

        is_worst = any(kw in q_lower for kw in ["underperforming", "lowest", "worst", "bottom", "cheapest"])

        # If user asked for specific records (e.g., "top 10 car in lounge" or "top 5 cars in pop")
        if (top_n_match or matched_filter_val) and not any(kw in q_lower for kw in ["highest sales", "which model", "total price", "chart"]):
            df_filtered = df.copy()
            if matched_filter_col and matched_filter_val:
                df_filtered = df_filtered[df_filtered[matched_filter_col].astype(str).str.lower() == matched_filter_val.lower()]

            if not df_filtered.empty:
                sort_col = target_num if target_num and target_num in df_filtered.columns else df_filtered.columns[0]
                try:
                    df_sorted = df_filtered.sort_values(by=sort_col, ascending=is_worst)
                except Exception:
                    df_sorted = df_filtered

                top_records = df_sorted.head(n_val).to_dict(orient="records")
                return {
                    "intent": "record_list",
                    "n": n_val,
                    "filter_col": matched_filter_col,
                    "filter_val": matched_filter_val,
                    "target_num": sort_col,
                    "is_worst": is_worst,
                    "records": top_records,
                    "total_matching": len(df_filtered)
                }

        # 2. ENTITY LOOKUP / LIST INTENT (e.g., "car lists", "list cars", "show models", "list categories", "column names")
        is_list_query = any(kw in q_lower for kw in ["list", "lists", "show all", "names", "models", "categories", "items"])

        if is_list_query and target_cat and target_cat in df.columns:
            counts = df[target_cat].value_counts().head(15).to_dict()
            return {
                "intent": "entity_list",
                "category_column": target_cat,
                "items_with_counts": counts,
                "total_unique": df[target_cat].nunique(),
                "sample_rows": df[[target_cat]].head(10).to_dict(orient="records")
            }

        if any(kw in q_lower for kw in ["how many column", "column names", "list columns", "fields"]):
            return {
                "intent": "column_list",
                "fields": list(df.columns),
                "total_columns": len(df.columns),
                "total_rows": len(df)
            }

        # 3. VISUALIZATION / CHART INTENT
        if "chart" in q_lower or "graph" in q_lower or "plot" in q_lower:
            if target_date and target_num:
                trend_df = self.tools.time_series_trend(df, target_date, target_num, freq="M")
                if not trend_df.empty:
                    return {
                        "intent": "chart",
                        "chart_type": "line",
                        "title": f"Monthly Trend of {target_num}",
                        "labels": trend_df["Period"].tolist()[:12],
                        "values": trend_df[f"Total {target_num}"].tolist()[:12]
                    }
            if target_cat and target_num:
                rank_df = self.tools.rank_group_by(df, target_cat, target_num, top_n=8)
                if not rank_df.empty:
                    return {
                        "intent": "chart",
                        "chart_type": "bar",
                        "title": f"Top {target_cat} by {target_num}",
                        "labels": rank_df[target_cat].astype(str).tolist(),
                        "values": rank_df[f"Total {target_num}"].tolist()
                    }

        # 4. FORECASTING / PREDICTION INTENT
        if any(kw in q_lower for kw in ["predict", "forecast", "project", "next month", "future"]):
            if target_date and target_num:
                fc = self.tools.run_forecasting(df, target_date, target_num, periods=3)
                if "error" in fc:
                    fc = self.tools.run_forecasting(self.df, target_date, target_num, periods=3)
                if "error" not in fc:
                    return {
                        "intent": "forecast",
                        "target_num": target_num,
                        "target_date": target_date,
                        "forecast_info": fc,
                        "chart_type": "line",
                        "labels": [f"Period +{r['period']}" for r in fc.get("forecast_results", [])],
                        "values": [r["projected_value"] for r in fc.get("forecast_results", [])]
                    }

        # 5. RANKING / TOP / BEST INTENT (AGGREGATED BY GROUP)
        if any(kw in q_lower for kw in ["top", "highest", "best", "performing", "underperforming", "lowest", "worst"]):
            if target_cat and target_num:
                rank_df = self.tools.rank_group_by(df, target_cat, target_num, top_n=10, ascending=is_worst)
                if not rank_df.empty:
                    return {
                        "intent": "ranking",
                        "is_worst": is_worst,
                        "target_cat": target_cat,
                        "target_num": target_num,
                        "ranking_table": rank_df.to_dict(orient="records"),
                        "chart_type": "bar",
                        "labels": rank_df[target_cat].astype(str).tolist()[:6],
                        "values": rank_df[f"Total {target_num}"].tolist()[:6]
                    }

        # 6. AGGREGATION / TOTAL / AVERAGE INTENT
        if any(kw in q_lower for kw in ["total", "sales", "revenue", "sum", "average", "avg", "mean", "count", "how many", "price"]):
            agg_fn = "mean" if any(w in q_lower for w in ["average", "avg", "mean"]) else "sum"
            if target_num:
                val = self.tools.aggregate_metric(df, target_num, agg_fn=agg_fn)
                return {
                    "intent": "aggregation",
                    "agg_fn": agg_fn,
                    "target_num": target_num,
                    "val": val,
                    "total_rows": len(df)
                }

        # Default Schema Breakdown
        return {
            "intent": "general",
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "sample": df.head(5).to_dict(orient="records")
        }

    def _call_generative_model(self, original_q: str, resolved_q: str, pandas_res: dict) -> dict:
        """
        Calls Generative Model (Gemini / OpenAI Proxy / ChatGPT / Ollama) or Smart Data Analyst Fallback.
        """
        active_count = len(self.active_df)
        filter_str = f" (Filtered: {', '.join(self.applied_filters)})" if self.applied_filters else ""

        sys_prompt = (
            "You are TARS, a natural AI Data Analyst like ChatGPT/Gemini.\n"
            "STRICT INSTRUCTIONS:\n"
            "1. Give ONLY a direct, natural, conversational response answering exactly what the user asked.\n"
            "2. Do NOT include unneeded headers like 'Explanation:', 'Key Insight:', or 'Strategic Recommendation:' unless requested.\n"
            "3. Ground your answer 100% in the provided dataset context and Pandas analysis results.\n"
            "4. Be clear, concise, accurate, and professional."
        )

        dataset_context = {
            "question": original_q,
            "filename": self.dataset_meta.get("filename", "uploaded_dataset"),
            "total_active_rows": active_count,
            "applied_filters": self.applied_filters,
            "schema_fields": list(self.df.columns),
            "pandas_analysis_result": pandas_res,
            "sample_data": self.active_df.head(10).to_dict(orient="records")
        }

        user_prompt = f"Dataset Analysis Result:\n{json.dumps(dataset_context, default=str)}\n\nUser Question: {original_q}\nProvide direct natural response:"

        # 1. Try Local OpenAI-compatible Gemini Proxy (main-openai.py on localhost:8000 or custom OPENAI_API_BASE)
        openai_base = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        openai_key = self.openai_key or os.getenv("OPENAI_API_KEY", "changeme_local_only")

        if HAS_OPENAI_SDK:
            try:
                client = openai.OpenAI(api_key=openai_key, base_url=openai_base)
                resp = client.chat.completions.create(
                    model="gemini-1.5-flash",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    timeout=5.0
                )
                text = resp.choices[0].message.content.strip()
                if text:
                    return self._format_ai_response(text, pandas_res)
            except Exception:
                pass

        # 2. Try Google Gemini API if Key is present
        if self.gemini_key:
            try:
                if HAS_GEMINI_SDK:
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    response = model.generate_content(f"{sys_prompt}\n\n{user_prompt}")
                    text = response.text.strip()
                    if text:
                        return self._format_ai_response(text, pandas_res)
                else:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_key}"
                    headers = {"Content-Type": "application/json"}
                    payload = json.dumps({
                        "contents": [{"parts": [{"text": f"{sys_prompt}\n\n{user_prompt}"}]}]
                    }).encode("utf-8")
                    req = urllib.request.Request(url, data=payload, headers=headers)
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        res_json = json.loads(resp.read().decode("utf-8"))
                        text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                        if text:
                            return self._format_ai_response(text, pandas_res)
            except Exception:
                pass

        # 3. Try Local Ollama SDK if running
        if HAS_OLLAMA_SDK:
            for model_name in ["qwen3:14b", "llama3.1:8b", "gemma3:4b", "mistral"]:
                try:
                    resp = ollama.chat(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        options={"temperature": 0.2}
                    )
                    text = resp["message"]["content"].strip()
                    if text:
                        return self._format_ai_response(text, pandas_res)
                except Exception:
                    continue

        # 4. Smart Direct Pandas Data Analyst Engine (Zero External AI Dependancy Fallback)
        return self._smart_direct_pandas_fallback(original_q, pandas_res)

    def _smart_direct_pandas_fallback(self, question: str, res: dict) -> dict:
        """Smart direct response fallback answering user questions concisely without forced cards."""
        q_lower = question.lower()
        intent = res.get("intent")
        meta = self.dataset_meta
        active_count = len(self.active_df)
        filter_str = f" (Filtered: {', '.join(self.applied_filters)})" if self.applied_filters else ""

        # Handle Specific Record List (e.g. "top 10 car in lounge", "top 5 records")
        if intent == "record_list":
            records = res["records"]
            n = res["n"]
            fcol = res.get("filter_col")
            fval = res.get("filter_val")
            tnum = res.get("target_num", "price")
            is_worst = res.get("is_worst", False)

            title_str = f"top {n}" if not is_worst else f"cheapest/lowest {n}"
            if fcol and fval:
                header = f"Here are the **{title_str}** records for **{fcol} = '{fval}'** (sorted by **{tnum}**):\n"
            else:
                header = f"Here are the **{title_str}** dataset records (sorted by **{tnum}**):\n"

            lines = [header]

            for idx, r in enumerate(records, 1):
                details = []
                for k, v in r.items():
                    if k.lower() in ["id", "model", "price", "engine_power", "km", "age_in_days", tnum.lower()]:
                        if isinstance(v, (int, float)) and ("price" in k.lower() or "revenue" in k.lower() or "amount" in k.lower()):
                            details.append(f"**{k}**: ${v:,.2f}")
                        elif isinstance(v, (int, float)):
                            details.append(f"**{k}**: {v:,}")
                        else:
                            details.append(f"**{k}**: {v}")
                lines.append(f"{idx}. " + " | ".join(details))

            direct = "\n".join(lines)
            followups = [f"What is average {tnum} for {fval or 'this dataset'}?", "Show monthly trend", "List column names"]
            return self._format_ai_response(direct, res, followups=followups)

        # Handle Entity Listing (e.g. "car lists", "list models", "show categories")
        if intent == "entity_list":
            cat_col = res["category_column"]
            counts = res["items_with_counts"]
            total_u = res["total_unique"]

            lines = [f"Here are the **{cat_col}** entries present in **{meta.get('filename', 'the dataset')}** ({total_u} unique values):\n"]
            for idx, (item, count) in enumerate(counts.items(), 1):
                pct = round((count / active_count) * 100, 1) if active_count > 0 else 0
                lines.append(f"{idx}. **{item}** – {count:,} records ({pct}%)")

            direct = "\n".join(lines)
            followups = [f"Show monthly trend for {list(counts.keys())[0]}", f"Compare top vs bottom {cat_col}", "What is total sales?"]
            
            # Auto Chart
            chart_data = {
                "type": "bar",
                "title": f"Distribution of {cat_col}",
                "labels": list(counts.keys())[:8],
                "values": list(counts.values())[:8]
            }
            return self._format_ai_response(direct, res, chart_data=chart_data, followups=followups)

        # Handle Column Listing
        if intent == "column_list":
            fields = res["fields"]
            direct = f"The dataset **{meta.get('filename', 'file')}** contains **{len(fields)} columns**:\n\n" + "\n".join([f"{i+1}. **{f}**" for i, f in enumerate(fields)])
            return self._format_ai_response(direct, res, followups=["Show top records", "Calculate total metrics", "Predict revenue"])

        # Handle Ranking (Top/Best)
        if intent == "ranking":
            target_cat = res["target_cat"]
            target_num = res["target_num"]
            rt = res["ranking_table"]
            top_item = rt[0][target_cat]
            top_val = rt[0][f"Total {target_num}"]
            is_worst = res.get("is_worst", False)

            direct = f"**{top_item}** is the {'lowest' if is_worst else 'highest'} performing **{target_cat}** with a total **{target_num}** of **{top_val:,.2f}** across {active_count:,} records{filter_str}."
            followups = [f"Show monthly trend for {top_item}", f"Compare top vs bottom {target_cat}", f"What is the average {target_num}?"]
            return self._format_ai_response(direct, res, followups=followups)

        # Handle Aggregations (Totals / Averages)
        if intent == "aggregation":
            target_num = res["target_num"]
            val = res["val"]
            agg_name = "Average" if res["agg_fn"] == "mean" else "Total"
            is_curr = any(k in target_num.lower() for k in ["price", "revenue", "sales", "amount", "cost", "salary", "value"])
            prefix = "$" if is_curr else ""
            
            user_label = "Sales (Price)" if ("sales" in q_lower or "revenue" in q_lower) and target_num.lower() == "price" else target_num
            direct = f"The calculated **{agg_name} {user_label}** is **{prefix}{val:,.2f}** across **{active_count:,} active records**{filter_str}."
            return self._format_ai_response(direct, res, followups=[f"Show top items by {target_num}", f"Break down {target_num} by category"])

        # Handle Forecast
        if intent == "forecast":
            fc = res["forecast_info"]
            results = fc.get("forecast_results", [])
            p1 = results[0]["projected_value"] if results else 0.0
            direct = f"Projected **{res['target_num']}** for the next period is **{p1:,.2f}** ({fc.get('trend_direction', 'Growth Trend')})."
            return self._format_ai_response(direct, res, followups=["Show historical trend line", "What is model accuracy?"])

        # Default Response
        direct = f"Analyzed active dataset containing **{active_count:,} rows** and **{len(self.df.columns)} columns**{filter_str}."
        return self._format_ai_response(direct, res)

    def _format_ai_response(self, text_answer: str, res: dict, chart_data: dict = None, followups: list = None) -> dict:
        """Formats clean output response object."""

        if not chart_data and "chart_type" in res:
            chart_data = {
                "type": res["chart_type"],
                "title": res.get("title", f"Visualization"),
                "labels": res.get("labels", []),
                "values": res.get("values", [])
            }

        default_followups = ["Show top categories", "Calculate total metric", "Predict future trend"]

        return {
            "direct_answer": text_answer,
            "chart_data": chart_data,
            "followups": followups if followups else default_followups,
            "active_rows": len(self.active_df),
            "applied_filters": self.applied_filters
        }
