import joblib
import numpy as np
import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, conlist
from typing import List

# --- OpenTelemetry Setup ---
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# 1. Set up a tracer provider
provider = TracerProvider()

# 2. Configure a processor and exporter (using ConsoleExporter for demonstration)
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)

# 3. Set the global tracer provider
trace.set_tracer_provider(provider)

# 4. Get a tracer for this application
tracer = trace.get_tracer(__name__)
# --- End OpenTelemetry Setup ---


# Initialize the FastAPI app
app = FastAPI(
    title="ML Model Deployment Service",
    description="A service to serve a scikit-learn model with health checks and tracing.",
    version="1.0.0"
)

# 5. Instrument the FastAPI app
# This will automatically create spans for all incoming requests.
FastAPIInstrumentor.instrument_app(app)


# --- Model Loading ---
# Try to load the model and store its state
try:
    model = joblib.load('model.joblib')  # Ensure this path is correct relative to your app.py
    model_ready = True
    print("Model loaded successfully.")
except Exception as e:
    model = None
    model_ready = False
    print(f"Error loading model: {e}")


# --- Pydantic Model for Input Validation ---
# Defines the structure and type of the request body for the /predict endpoint
class PredictionRequest(BaseModel):
    features: List[conlist(item_type=float, min_length=4, max_length=4)] # Example: assuming 4 features

class PredictionResponse(BaseModel):
    prediction: list


# --- API Endpoints ---

@app.get('/')
def home():
    """Home endpoint to test if the service is accessible."""
    return {"message": "Model deployment service is running."}


@app.post('/predict', response_model=PredictionResponse)
def predict(request_data: PredictionRequest):
    """Endpoint to make predictions using the loaded model."""
    # The FastAPIInstrumentor automatically creates a span. We can get the current span.
    span = trace.get_current_span()

    if not model_ready or model is None:
        span.set_attribute("model.ready", "false")
        # Use HTTPException for standard FastAPI error handling
        raise HTTPException(status_code=503, detail="Model is not loaded or not ready")

    try:
        # Pydantic has already validated the input data
        features = np.array(request_data.features)
        span.set_attribute("features.shape", str(features.shape))

        # Make prediction
        prediction = model.predict(features)

        # Return prediction
        response = {'prediction': prediction.tolist()}
        span.set_attribute("prediction.result", str(response))

        return response
    except Exception as e:
        # Record the exception in the span and raise an HTTP exception
        span.record_exception(e)
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        raise HTTPException(status_code=400, detail=str(e))


# --- Health Check Endpoints for Kubernetes ---

@app.get('/live', status_code=200)
def is_live():
    """
    Liveness probe endpoint.
    Tells Kubernetes if the application is running (alive).
    """
    return {"status": "live"}


@app.get('/ready', status_code=200)
def is_ready():
    """
    Readiness probe endpoint.
    Tells Kubernetes if the application is ready to accept traffic.
    """
    if model_ready:
        return {"status": "ready"}
    else:
        # Return a 503 Service Unavailable status code if not ready
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "Model not loaded"}
        )

# Note: To run this app locally, use: uvicorn app:app --reload
# In the container, you would run: uvicorn app:app --host 0.0.0.0 --port 8080
