"""Local-only, dependency-light web demo for live presentation & testing.
Run from the project root:
    python app.py
Then open http://127.0.0.1:7860 in your browser.
"""
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parent

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_LABELS = ["0-12m", "13-24m", "25-48m", "49-72m"]
REQUIRED_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "PaymentMethod", "Contract", *SERVICE_COLUMNS]
ENGINEERED_COLUMNS = [
    "AvgMonthlySpend", "PriceHikeRatio", "TotalServicesCount",
    "IsAutoPayment", "HasHighRiskContract", "TenureCohort",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless row-wise feature engineering (matches model_building.ipynb)."""

    def __init__(self, epsilon: float = 1e-5, neutral_ratio_at_zero_tenure: float = 1.0):
        self.epsilon = epsilon
        self.neutral_ratio_at_zero_tenure = neutral_ratio_at_zero_tenure

    def fit(self, X: pd.DataFrame, y=None):
        self._validate(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        expected = getattr(self, "feature_names_in_", None)
        df = X.loc[:, list(expected)].copy() if expected is not None else X.copy()

        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["PriceHikeRatio"] = df["MonthlyCharges"] / (df["AvgMonthlySpend"] + self.epsilon)
        if self.neutral_ratio_at_zero_tenure is not None:
            df.loc[df["tenure"] == 0, "PriceHikeRatio"] = self.neutral_ratio_at_zero_tenure
        df["TotalServicesCount"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1).astype(int)
        df["IsAutoPayment"] = df["PaymentMethod"].astype(str).str.contains("automatic", case=False, na=False).astype(int)
        df["HasHighRiskContract"] = (df["Contract"] == "Month-to-month").astype(int)
        df["TenureCohort"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, right=True).astype(str)
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + ENGINEERED_COLUMNS, dtype=object)

    @staticmethod
    def _validate(X: pd.DataFrame):
        missing = [c for c in REQUIRED_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


MODELS = {
    "SVM": joblib.load(ROOT / "customer_churn_model.pkl"),
    "KNN": joblib.load(ROOT / "customer_churn_knn_model.pkl"),
}

METRICS = {
    "KNN": {"test_metrics": {"accuracy": 0.7906, "precision": 0.6106, "recall": 0.5829, "f1": 0.5964, "roc_auc": 0.8391}},
    "SVM": {"test_metrics": {"accuracy": 0.7502, "precision": 0.5189, "recall": 0.8075, "f1": 0.6318, "roc_auc": 0.8277}},
}

OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["Fiber optic", "DSL", "No"],
    **{x: ["No", "Yes", "No internet service"] for x in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]},
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ],
}

DEFAULTS = {k: v[0] for k, v in OPTIONS.items()}
DEFAULTS.update(SeniorCitizen=0, tenure=14, MonthlyCharges=79.35, TotalCharges=1067.8)

HTML_PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Churn Lab | Group 2</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#13243a;font:15px system-ui,sans-serif}header{background:#13243a;color:white;padding:32px max(24px,calc((100% - 1200px)/2))}header small{color:#75dac1;font-weight:700;letter-spacing:2px}h1{font-size:34px;margin:10px 0}header p{color:#ced9e7;max-width:780px;line-height:1.6}main{max-width:1248px;margin:28px auto;padding:0 24px;display:grid;grid-template-columns:1fr 330px;gap:24px}.card{background:white;border:1px solid #dce3ec;border-radius:16px;padding:24px;margin-bottom:20px}h2{margin:0 0 8px;font-size:21px}.hint{color:#5b6a7c;line-height:1.6;font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}label{display:block;color:#42516a;font-size:13px;font-weight:600}input,select{display:block;width:100%;padding:11px;border:1px solid #c9d2df;border-radius:8px;margin-top:6px;color:#13243a;background:#fff;font:inherit}button{background:#116959;color:white;border:0;border-radius:8px;padding:13px 20px;cursor:pointer;font:600 15px system-ui}button.alt{background:#e7eeef;color:#16483f}fieldset{border:0;padding:0;margin:24px 0}legend{font-weight:700;border-bottom:1px solid #e2e7ee;width:100%;padding-bottom:9px;margin-bottom:14px}.big{font-size:28px;font-weight:750;margin:16px 0}.score{background:#eef7f4;padding:16px;border-radius:9px;margin:16px 0}#error{color:#a52232;line-height:1.6;margin:14px 0}dl{display:grid;grid-template-columns:1fr auto;gap:12px;font-size:14px}dt,dd{margin:0}.pill{display:inline-block;background:#e8f4ef;color:#116959;border-radius:20px;padding:5px 10px;font-size:12px;font-weight:700}footer{font-size:12px;color:#627188;line-height:1.7}@media(max-width:850px){main{grid-template-columns:1fr}.grid{grid-template-columns:1fr 1fr}header{padding:25px}h1{font-size:28px}}@media(max-width:480px){.grid{grid-template-columns:1fr}}</style>
<header><small>GROUP 2 / MODELING & EVALUATION</small><h1>Customer Churn Lab</h1><p>Test KNN and SVM on a new customer profile. The app applies the training preprocessing pipeline automatically, then returns the selected model's prediction.</p></header>
<main><section><form id="form" class="card"><div class="pill">Local demo · Inputs are not saved</div><h2 style="margin-top:16px">New customer profile</h2><p class="hint">Enter raw customer fields, not scaled values. Values outside the training population may be less reliable.</p><label>Prediction model<select id="model"><option>SVM</option><option>KNN</option></select></label><div id="fields"></div><div id="error" role="alert"></div><button id="submit" type="submit">Predict churn</button> <button class="alt" id="reset" type="button">Reset example</button><p class="hint">The prefilled profile is a synthetic input example.</p></form></section>
<aside><div class="card" aria-live="polite"><div class="pill">Prediction</div><div id="result"><p class="big">Ready to test</p><p class="hint">Fill the form and choose Predict churn. Results will appear here.</p></div></div><div class="card"><h2>Held-out performance</h2><p class="hint">Same 1,409 test customers (5-fold CV on training data only).</p><div id="metrics"></div></div><footer>Churn: 1 = leaves, 0 = stays.<br>SVM margin is not a probability. KNN vote share is not calibrated.<br><br>Run locally at 127.0.0.1.</footer></aside></main>
<script>
let config;const names={SeniorCitizen:'Senior citizen (0 = No, 1 = Yes)',tenure:'Tenure (months)',MonthlyCharges:'Monthly charges',TotalCharges:'Total charges',gender:'Gender',Partner:'Partner',Dependents:'Dependents',PhoneService:'Phone service',MultipleLines:'Multiple lines',InternetService:'Internet service',OnlineSecurity:'Online security',OnlineBackup:'Online backup',DeviceProtection:'Device protection',TechSupport:'Tech support',StreamingTV:'Streaming TV',StreamingMovies:'Streaming movies',Contract:'Contract',PaperlessBilling:'Paperless billing',PaymentMethod:'Payment method'};
const groups={'Customer':['gender','SeniorCitizen','Partner','Dependents'],'Account & billing':['tenure','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges'],'Connected services':['PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']};
function defaults(){for(let [k,v] of Object.entries(config.defaults))document.getElementById(k).value=v;document.getElementById('error').textContent=''}
function metric(){const key=document.getElementById('model').value,m=config.metrics[key].test_metrics;document.getElementById('metrics').innerHTML='<b>'+key+'</b><dl>'+['accuracy','precision','recall','f1','roc_auc'].map(k=>'<dt>'+k.replace('_',' ').toUpperCase()+'</dt><dd>'+m[k].toFixed(3)+'</dd>').join('')+'</dl>'}
fetch('/api/config').then(r=>r.json()).then(c=>{config=c;let html='';for(let [group,fields]of Object.entries(groups)){html+='<fieldset><legend>'+group+'</legend><div class="grid">';for(let k of fields){html+='<label>'+names[k];if(c.options[k])html+='<select required id="'+k+'">'+c.options[k].map(v=>'<option>'+v+'</option>').join('')+'</select>';else html+='<input required id="'+k+'" type="number" min="0" '+(k==='tenure'?'max="72" step="1"':k==='SeniorCitizen'?'max="1" step="1"':'step="0.01"')+'>';html+='</label>'}html+='</div></fieldset>'}document.getElementById('fields').innerHTML=html;defaults();metric()});
document.getElementById('reset').onclick=defaults;document.getElementById('model').onchange=metric;
document.getElementById('form').onsubmit=async e=>{e.preventDefault();const btn=document.getElementById('submit');btn.disabled=true;document.getElementById('error').textContent='';try{const customer={};for(let k of Object.keys(config.defaults))customer[k]=document.getElementById(k).value;const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:document.getElementById('model').value,customer})});const r=await response.json();if(!response.ok)throw Error(r.error);const box=document.getElementById('result');box.replaceChildren();const label=document.createElement('p');label.className='big';label.textContent=r.label;const model=document.createElement('p');model.className='hint';model.textContent=r.model+' · Class '+r.prediction;const score=document.createElement('div');score.className='score';score.textContent=r.score_label+': '+r.score.toFixed(4);const note=document.createElement('p');note.className='hint';note.textContent=r.note;box.append(label,model,score,note)}catch(err){document.getElementById('error').textContent=err.message}finally{btn.disabled=false}};
</script></html>"""


def coerce_senior_citizen(value):
    if isinstance(value, bool):
        raise ValueError("SeniorCitizen must be 0 or 1.")
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return float(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return float(value)
    if isinstance(value, str) and value in ("0", "1"):
        return float(value)
    raise ValueError("SeniorCitizen must be 0 or 1 ('0'/'1' or 0.0/1.0 also accepted).")


def predict(payload):
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object.")
    key = payload.get("model", "SVM")
    if key not in MODELS:
        raise ValueError("Choose KNN or SVM.")
    data = payload.get("customer", {})
    if not isinstance(data, dict):
        raise ValueError("customer must be a JSON object.")
    missing = set(DEFAULTS) - set(data)
    if missing:
        raise ValueError("Missing fields: " + ", ".join(sorted(missing)))

    row = {k: data[k] for k in DEFAULTS}
    for col, choices in OPTIONS.items():
        if row[col] not in choices:
            raise ValueError(f"Invalid value for {col}.")
    row["SeniorCitizen"] = coerce_senior_citizen(row["SeniorCitizen"])

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        try:
            row[col] = float(row[col])
        except (TypeError, ValueError):
            raise ValueError(f"{col} must be a number.") from None
        if not np.isfinite(row[col]):
            raise ValueError(f"{col} must be finite.")

    if not 0 <= row["tenure"] <= 72 or not row["tenure"].is_integer():
        raise ValueError("Tenure must be a whole number from 0 to 72 months.")
    if min(row["MonthlyCharges"], row["TotalCharges"]) < 0:
        raise ValueError("Charges cannot be negative.")
    if row["tenure"] == 0 and row["TotalCharges"] != 0:
        raise ValueError("For zero tenure, total charges must be zero.")
    if (row["PhoneService"] == "No") != (row["MultipleLines"] == "No phone service"):
        raise ValueError("Multiple lines must be No phone service exactly when Phone service is No.")
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        if (row["InternetService"] == "No") != (row[col] == "No internet service"):
            raise ValueError(f"Check {col}: use No internet service exactly when Internet service is No.")

    row["tenure"] = int(row["tenure"])
    X = pd.DataFrame([row])
    pipe = MODELS[key]
    label = int(pipe.predict(X)[0])
    result = {
        "model": key,
        "prediction": label,
        "label": "Likely churn" if label else "Likely stay",
        "note": "Model estimate, not a guarantee. This demo does not save customer inputs.",
    }
    if key == "KNN":
        result.update(score=float(pipe.predict_proba(X)[0, 1]), score_label="Churn vote share (not calibrated probability)")
    else:
        result.update(score=float(pipe.decision_function(X)[0]), score_label="SVM decision margin (not probability)")
    return result


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body, ctype = HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8"
        elif self.path == "/api/config":
            body = json.dumps({"options": OPTIONS, "defaults": DEFAULTS, "metrics": METRICS}).encode()
            ctype = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404)
            return
        try:
            length = self.headers.get("Content-Length")
            try:
                size = int(length)
            except (TypeError, ValueError):
                raise ValueError("Content-Length must be a non-negative integer.") from None
            if not 0 <= size < 16384:
                raise ValueError("Invalid input size.")
            result = predict(json.loads(self.rfile.read(size)))
            status = 200
        except (ValueError, TypeError, KeyError) as exc:
            result = {"error": str(exc)}
            status = 400
        body = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    print(f"Open http://127.0.0.1:{args.port} in your browser. Press Ctrl+C to stop.", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
