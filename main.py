from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import logging
from contextlib import asynccontextmanager
from database import connect_to_mongo, close_mongo_connection, log_prediction, get_recent_predictions, get_prediction_stats
from model import get_feature_importances, train_and_save_model, MODEL_PATH

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model globally
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    
    global model
    if not os.path.exists(MODEL_PATH):
        logger.info("Model not found. Training a new model...")
        train_and_save_model()
    
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
        
    yield
    
    # Shutdown
    await close_mongo_connection()

app = FastAPI(
    title="Salary Prediction API",
    description="API for predicting employee salaries using an ML pipeline.",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    experience: int = Field(..., ge=0, description="Years of experience")
    leaves: int = Field(..., ge=0, description="Number of leaves taken")
    working_hours_per_day: float = Field(..., ge=1.0, le=24.0, description="Working hours per day")
    degree: str = Field(..., description="Highest degree obtained (e.g., Bachelors, Masters, PhD)")

class PredictionResponse(BaseModel):
    predicted_salary: float
    currency: str = "INR"

@app.post("/predict", response_model=PredictionResponse)
async def predict_salary(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Predict salary based on experience, leaves, and degree.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        # Prepare input data as a DataFrame for the pipeline
        input_df = pd.DataFrame([{
            'experience': request.experience,
            'leaves': request.leaves,
            'working_hours_per_day': request.working_hours_per_day,
            'degree': request.degree
        }])
        
        # Predict
        prediction = float(model.predict(input_df)[0])
        
        # Log to DB asynchronously in the background
        input_data = request.model_dump()
        background_tasks.add_task(log_prediction, input_data, prediction)
        
        return PredictionResponse(predicted_salary=round(prediction, 2))
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance")
def get_importance():
    """
    Returns feature importances from the trained model.
    """
    try:
        importances = get_feature_importances()
        return {"importances": importances}
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/predictions")
async def get_predictions(limit: int = 5):
    """Retrieve recent predictions."""
    try:
        preds = await get_recent_predictions(limit)
        return preds
    except Exception as e:
        logger.error(f"Failed to retrieve predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/stats")
async def get_stats():
    """Retrieve database prediction stats."""
    try:
        stats = await get_prediction_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to retrieve predictions stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def retrain_model():
    """Retrain the Random Forest model."""
    try:
        train_and_save_model()
        # Reload the model
        global model
        model = joblib.load(MODEL_PATH)
        logger.info("Model retrained and loaded successfully.")
        return {"status": "success", "message": "Model retrained successfully."}
    except Exception as e:
        logger.error(f"Failed to retrain model: {e}")
        raise HTTPException(status_code=500, detail=str(e))
