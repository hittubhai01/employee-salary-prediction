# Salary Prediction System

A salary prediction platform built with FastAPI, Streamlit, and MongoDB. The repository contains a backend prediction API, a frontend dashboard, and a machine learning model training pipeline.

## Features

- FastAPI backend for salary prediction and statistics
- Streamlit frontend dashboard for interacting with the prediction engine
- RandomForest-based salary prediction model
- MongoDB logging for prediction history and stats
- Docker Compose setup for easy local deployment

## Tech Stack

- Python 3.10
- FastAPI
- Streamlit
- MongoDB
- scikit-learn
- Docker / Docker Compose

## Requirements

- Python 3.10+
- Docker and Docker Compose (recommended)
- Local MongoDB if running without Docker

## Install Dependencies

```bash
cd /Users/hiteshyadav/Desktop/salary_prediction_system
source venv/bin/activate
pip install -r requirements.txt
```

## Run with Docker Compose (recommended)

```bash
cd /Users/hiteshyadav/Desktop/salary_prediction_system
docker compose up --build
```

Open the application in your browser:

- Streamlit UI: `http://localhost:8501`
- Backend API: `http://localhost:8000`

## Run Locally Without Docker

1. Activate the virtual environment:

```bash
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start MongoDB locally or configure `MONGO_URL`:

```bash
export MONGO_URL="mongodb://localhost:27017"
```

4. Start the FastAPI backend:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

5. Start the Streamlit UI:

```bash
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

Then open `http://localhost:8501`.

## API Endpoints

- `POST /predict` - Predict salary based on experience, leaves, working hours, and degree
- `GET /feature-importance` - Get feature importances from the trained model
- `GET /health` - Health check endpoint
- `GET /predictions` - Retrieve recent predictions
- `GET /predictions/stats` - Get aggregate prediction statistics
- `POST /train` - Retrain the model and save the updated pipeline

## Environment Variables

- `MONGO_URL` - MongoDB connection string
- `API_URL` - Used by the Streamlit app if running separately from the backend

## Project Structure

- `app.py` - Streamlit UI application
- `main.py` - FastAPI backend application
- `database.py` - MongoDB connection and logging utilities
- `model.py` - Model training and feature importance functions
- `requirements.txt` - Python dependencies
- `docker-compose.yml` - Docker Compose services for the app
- `Dockerfile.api` - Backend Docker image build configuration
- `Dockerfile.ui` - Frontend Docker image build configuration

## Notes

- The backend will auto-train the model and create `model.pkl` if it does not exist.
- The Docker Compose setup includes a MongoDB container and configures the UI to connect to the backend.
- If the backend cannot reach MongoDB, the API may still start, but logging and stats endpoints will be limited.
