import os
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use environment variable for MongoDB URL, default to localhost for local testing
MONGO_URL = os.getenv("MONGO_URL", "mongodb+srv://yhitesh702_db_user:jlbZOkb0vhq2Up01@clustersalary.hyf6m5g.mongodb.net/?appName=Clustersalary")
DB_NAME = "salary_prediction_db"
COLLECTION_NAME = "predictions"

class Database:
    client: AsyncIOMotorClient = None

db = Database()

async def connect_to_mongo():
    """Establish connection to MongoDB."""
    try:
        logger.info(f"Connecting to MongoDB at {MONGO_URL}...")
        db.client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        # Test connection
        await db.client.server_info()
        logger.info("Successfully connected to MongoDB!")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        # Not exiting here to allow the API to start even if DB is down initially
        # The app should handle DB insertion failures gracefully

async def close_mongo_connection():
    """Close MongoDB connection."""
    if db.client is not None:
        logger.info("Closing MongoDB connection...")
        db.client.close()
        logger.info("MongoDB connection closed.")

async def log_prediction(input_data: dict, prediction: float):
    """
    Log a prediction request and result to MongoDB.
    
    Args:
        input_data (dict): The user inputs (experience, leaves, degree).
        prediction (float): The predicted salary.
    """
    if db.client is None:
        logger.warning("MongoDB client is not initialized. Prediction log skipped.")
        return

    try:
        database = db.client[DB_NAME]
        collection = database[COLLECTION_NAME]
        
        record = {
            "input": input_data,
            "prediction": prediction,
            "timestamp": datetime.utcnow()
        }
        
        result = await collection.insert_one(record)
        logger.info(f"Successfully logged prediction. Document ID: {result.inserted_id}")
    except Exception as e:
        logger.error(f"Failed to log prediction to MongoDB: {e}")

async def get_recent_predictions(limit: int = 5):
    """Retrieve recent predictions from MongoDB."""
    if db.client is None:
        logger.warning("MongoDB client is not initialized. Cannot retrieve predictions.")
        return []
    try:
        database = db.client[DB_NAME]
        collection = database[COLLECTION_NAME]
        cursor = collection.find().sort("timestamp", -1).limit(limit)
        results = await cursor.to_list(length=limit)
        
        for record in results:
            record["_id"] = str(record["_id"])
            if "timestamp" in record and record["timestamp"]:
                record["timestamp"] = record["timestamp"].isoformat()
        return results
    except Exception as e:
        logger.error(f"Failed to fetch recent predictions from MongoDB: {e}")
        return []

async def get_prediction_stats():
    """Retrieve aggregate statistics for logged predictions."""
    if db.client is None:
        return {"count": 0, "avg_salary": 0.0}
    try:
        database = db.client[DB_NAME]
        collection = database[COLLECTION_NAME]
        
        count = await collection.count_documents({})
        if count == 0:
            return {"count": 0, "avg_salary": 0.0}
            
        pipeline = [
            {"$group": {"_id": None, "avg_salary": {"$avg": "$prediction"}}}
        ]
        cursor = collection.aggregate(pipeline)
        result = await cursor.to_list(length=1)
        
        avg_salary = result[0]["avg_salary"] if result else 0.0
        return {"count": count, "avg_salary": round(avg_salary, 2)}
    except Exception as e:
        logger.error(f"Failed to fetch prediction stats from MongoDB: {e}")
        return {"count": 0, "avg_salary": 0.0}
