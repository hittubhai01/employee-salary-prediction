import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

MODEL_PATH = "model.pkl"

def generate_synthetic_data(n_samples=1000):
    """Generates a synthetic dataset for salary prediction."""
    np.random.seed(42)
    
    # Features
    experience = np.random.randint(0, 30, n_samples)
    leaves = np.random.randint(0, 30, n_samples)
    working_hours_per_day = np.random.uniform(6, 12, n_samples)
    degrees = ['Bachelors', 'Masters', 'PhD']
    degree = np.random.choice(degrees, n_samples, p=[0.6, 0.3, 0.1])
    
    # Base salary calculation
    base_salary = 50000
    experience_multiplier = 2000
    leaves_penalty = -500
    hours_multiplier = 5000
    
    degree_bonus = {'Bachelors': 0, 'Masters': 15000, 'PhD': 30000}
    
    salary = base_salary + (experience * experience_multiplier) + (leaves * leaves_penalty) + ((working_hours_per_day - 8) * hours_multiplier)
    salary = salary + np.array([degree_bonus[d] for d in degree])
    
    # Add some noise
    noise = np.random.normal(0, 5000, n_samples)
    salary = salary + noise
    
    df = pd.DataFrame({
        'experience': experience,
        'leaves': leaves,
        'working_hours_per_day': working_hours_per_day,
        'degree': degree,
        'salary': salary
    })
    
    return df

def train_and_save_model():
    """Trains the ML pipeline and saves it to a file."""
    print("Generating synthetic data...")
    df = generate_synthetic_data()
    
    X = df[['experience', 'leaves', 'working_hours_per_day', 'degree']]
    y = df['salary']
    
    # Define preprocessing steps
    numeric_features = ['experience', 'leaves', 'working_hours_per_day']
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_features = ['degree']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the complete pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    print("Training model...")
    model.fit(X, y)
    
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Model training complete.")

def get_feature_importances():
    """Extracts feature importances from the trained pipeline."""
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
        
    model = joblib.load(MODEL_PATH)
    
    # Get feature names after one-hot encoding
    preprocessor = model.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_encoder.get_feature_names_out(['degree'])
    
    feature_names = ['experience', 'leaves', 'working_hours_per_day'] + list(cat_features)
    importances = model.named_steps['regressor'].feature_importances_
    
    # Clean up feature names for the frontend
    clean_names = []
    for name in feature_names:
        if name.startswith('degree_'):
            clean_names.append(name.replace('degree_', 'Degree: '))
        else:
            clean_names.append(name.capitalize())
            
    return dict(zip(clean_names, importances))

if __name__ == "__main__":
    train_and_save_model()
