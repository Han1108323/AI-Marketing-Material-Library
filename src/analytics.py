import json
import pandas as pd
import numpy as np
from typing import List, Dict

def _parse_multi(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        return [x.strip() for x in value.replace("，", ",").split(",") if x.strip()]
    return [str(value)]

def aggregate_ctr_by_tag(points, field="visual_style"):
    rows = []
    for p in points:
        payload = p.payload
        ctr = payload.get("historical_ctr", 0.0)
        tags = payload.get("tags", {})
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                tags = {}
        styles = tags.get(field, "")
        for t in _parse_multi(styles):
            rows.append({"tag": t, "ctr": ctr})
    if not rows:
        return pd.DataFrame(columns=["tag", "avg_ctr", "count"])
    df = pd.DataFrame(rows)
    agg = df.groupby("tag")["ctr"].agg(["mean", "count"]).reset_index()
    agg.columns = ["tag", "avg_ctr", "count"]
    agg = agg.sort_values("avg_ctr", ascending=False)
    return agg

def derive_supply_refs(points, high_threshold=0.04, low_threshold=0.02, field="visual_style", top_n=10):
    high, low = [], []
    for p in points:
        payload = p.payload
        ctr = payload.get("historical_ctr", 0.0)
        tags = payload.get("tags", {})
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except:
                tags = {}
        styles = _parse_multi(tags.get(field, ""))
        if ctr is None:
            continue
        if ctr >= high_threshold:
            high.extend(styles)
        elif ctr <= low_threshold:
            low.extend(styles)
    def top_list(ls):
        import collections
        c = collections.Counter(ls)
        return [t for t, _ in c.most_common(top_n)]
    positive = top_list(high)
    negative = top_list(low)
    recommended = [t for t in positive if t not in set(negative)]
    return {
        "positive_refs": positive,
        "negative_refs": negative,
        "recommended_tags": recommended
    }

def run_validation(materials: List[Dict]) -> Dict:
    """
    Backtesting validation: Compare predicted CTR vs Real CTR.
    materials: List of dicts with keys ['material_id', 'predicted_ctr', 'real_ctr']
    """
    if not materials:
        return {"error": "No data for validation"}
    
    df = pd.DataFrame(materials)
    
    # Ensure columns exist
    if 'predicted_ctr' not in df.columns or 'real_ctr' not in df.columns:
        return {"error": "Missing required columns: predicted_ctr, real_ctr"}
        
    # Drop N/A
    df = df.dropna(subset=['predicted_ctr', 'real_ctr'])
    
    if len(df) < 2:
        return {"error": "Insufficient data points (need >= 2)"}
        
    # Calculate Correlation
    correlation = df['predicted_ctr'].corr(df['real_ctr'])
    
    # Interpretation
    if abs(correlation) > 0.7:
        interpretation = "strong"
    elif abs(correlation) > 0.4:
        interpretation = "moderate"
    else:
        interpretation = "weak"
        
    # Calculate Error
    df['error'] = df['predicted_ctr'] - df['real_ctr']
    mae = df['error'].abs().mean()
    mse = (df['error'] ** 2).mean()
    
    # Detailed Comparison
    comparison_data = []
    for _, row in df.iterrows():
        comparison_data.append({
            "material_id": row.get('material_id', 'unknown'),
            "predicted": row['predicted_ctr'],
            "real": row['real_ctr'],
            "error": round(row['error'], 4)
        })
        
    return {
        "correlation": round(correlation, 4) if not np.isnan(correlation) else 0.0,
        "interpretation": interpretation,
        "sample_count": len(df),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "comparison_data": comparison_data[:50] # Limit return size
    }
