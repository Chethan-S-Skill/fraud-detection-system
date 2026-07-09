import os
import sys
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Summary, make_asgi_app

# Add src to python path to load our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom ML pipeline elements
try:
    from train import train_model
    from explain import FraudSHAPExplainer
    from preprocessing import FraudFeatureExtractor, PipelinePreprocessor
    from chatbot import ShieldAICopilot
except ImportError as e:
    print(f"Error importing modules: {e}")

# Initialize FastAPI App
app = FastAPI(
    title="ShieldAI - Enterprise Fraud Detection API",
    description="Real-time transaction risk scoring with SHAP explainability and Prometheus monitoring.",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Prometheus Monitoring Metrics
TRANSACTION_COUNT = Counter(
    'shieldai_transactions_total',
    'Total number of transactions processed.',
    ['merchant_category', 'classification']
)
FRAUD_PROBABILITY_SUMMARY = Summary(
    'shieldai_fraud_probability',
    'Summary of fraud risk probabilities predicted by the model.'
)
INFERENCE_LATENCY = Histogram(
    'shieldai_inference_latency_seconds',
    'Latency of transaction inference in seconds.'
)

# Model paths
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_model.joblib")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
EXTRACTOR_PATH = os.path.join(MODELS_DIR, "feature_extractor.joblib")

# Global containers loaded on startup
model = None
preprocessor = None
extractor = None
shap_explainer = None

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_4569871")
    timestamp: str = Field(..., example="2026-07-09T10:00:00")
    user_id: str = Field(..., example="USR_0124")
    amount: float = Field(..., example=450.50, ge=0.0)
    merchant_category: str = Field(..., example="e_commerce")
    user_lat: float = Field(..., example=37.7749)
    user_lon: float = Field(..., example=-122.4194)
    distance_from_home: float = Field(..., example=15.4, ge=0.0)
    device_id: str = Field(..., example="DEV_8832")
    card_present: int = Field(..., example=0, le=1, ge=0)
    threshold: float = Field(default=0.5, example=0.5, le=1.0, ge=0.0)

class ChatRequest(BaseModel):
    message: str = Field(..., example="explain latest transaction")

@app.on_event("startup")
def startup_event():
    global model, preprocessor, extractor, shap_explainer, copilot
    
    # Auto-train model if it does not exist
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Auto-generating data and training model...")
        train_model(data_path="data/transactions.csv", models_dir=MODELS_DIR, num_trials=5)
        
    print("Loading models and preprocess pipelines...")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    extractor = joblib.load(EXTRACTOR_PATH)
    shap_explainer = FraudSHAPExplainer(MODEL_PATH)
    copilot = ShieldAICopilot()
    print("Models and pipelines successfully loaded!")

@app.post("/predict", status_code=200)
async def predict_fraud(txn: TransactionRequest):
    global model, preprocessor, extractor, shap_explainer
    
    if model is None or preprocessor is None or extractor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please wait or check logs."
        )
        
    start_time = time.time()
    
    try:
        # 1. Real-time Feature Extraction (rolling windows + device mismatch flags)
        txn_dict = txn.dict()
        # Extract features & update user transaction history cache
        engineered_txn = extractor.extract_single_feature(txn_dict, update_history=True)
        
        # 2. Pipeline Preprocessing
        # Convert single item to df matching input structure
        raw_df = pd.DataFrame([engineered_txn])
        
        # Drop columns not used by the ML model
        non_feature_cols = ['transaction_id', 'timestamp', 'user_id', 'device_id', 'threshold']
        feature_df = raw_df.drop(columns=[col for col in non_feature_cols if col in raw_df.columns])
        
        # Run preprocessing transform
        preprocessed_df = preprocessor.transform(feature_df)
        
        # 3. Model Prediction
        risk_prob = float(model.predict_proba(preprocessed_df)[0, 1])
        
        # Classification decision based on requested threshold
        is_fraud = risk_prob >= txn.threshold
        classification = "Blocked" if is_fraud else "Approved"
        
        # 4. Generate SHAP explainability contributions
        explanation = shap_explainer.explain_transaction(preprocessed_df)
        
        # Compute latency
        latency = time.time() - start_time
        
        # 5. Log metrics to Prometheus
        TRANSACTION_COUNT.labels(
            merchant_category=txn.merchant_category,
            classification=classification
        ).inc()
        FRAUD_PROBABILITY_SUMMARY.observe(risk_prob)
        INFERENCE_LATENCY.observe(latency)
        
        # Build friendly engineered features output
        features_output = {
            "txn_count_1h": int(engineered_txn["txn_count_1h"]),
            "spend_sum_1h": float(engineered_txn["spend_sum_1h"]),
            "txn_count_24h": int(engineered_txn["txn_count_24h"]),
            "spend_sum_24h": float(engineered_txn["spend_sum_24h"]),
            "is_new_device": int(engineered_txn["is_new_device"]),
            "hour_of_day": int(engineered_txn["hour_of_day"]),
            "day_of_week": int(engineered_txn["day_of_week"]),
        }
        
        return {
            "transaction_id": txn.transaction_id,
            "risk_score": round(risk_prob, 4),
            "is_fraud": bool(is_fraud),
            "classification": classification,
            "latency_ms": round(latency * 1000, 2),
            "features_engineered": features_output,
            "shap_explanations": {
                "base_probability": round(explanation["base_probability"], 4),
                "contributions": explanation["contributions"]
            }
        }
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failed: {str(e)}"
        )

@app.post("/chat", status_code=200)
async def chat_copilot(request: ChatRequest):
    global copilot
    if copilot is None:
        copilot = ShieldAICopilot()
    try:
        response = copilot.respond(request.message)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chatbot failed to process message: {str(e)}"
        )

@app.get("/health", status_code=200)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "extractor_loaded": extractor is not None,
        "timestamp": time.time()
    }

# Mount Prometheus ASGI app to expose standard Prometheus metrics at /metrics
app.mount("/metrics", make_asgi_app())

# Mount frontend web client static files
# Place web directory in root folder
web_path = os.path.join(os.path.dirname(__file__), "web")
if os.path.exists(web_path):
    app.mount("/", StaticFiles(directory=web_path, html=True), name="web")
else:
    print(f"Warning: web frontend directory not found at {web_path}. Serving API only.")
