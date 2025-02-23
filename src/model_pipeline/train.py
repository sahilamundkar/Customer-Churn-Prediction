# src/model_pipeline/train.py
import pandas as pd
from typing import Tuple, Dict 
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, feature_path: str, model_path: str):
        self.feature_path = Path(feature_path)
        self.model_path = Path(model_path)
        
        # Features to use (based on our previous analysis)
        self.features = [
            'bounce_rate', 'pages_per_session',
            'sessions', 'pageviews', 'dsls',
            'visited_demo_page', 'visited_water_purifier_page',
            'visited_vacuum_cleaner_page',
            'help_me_buy_evt_count', 'phone_clicks_evt_count'
        ]
        
    def load_latest_features(self) -> pd.DataFrame:
        """Load the most recent feature set"""
        feature_files = list(self.feature_path.glob("features_*.csv"))
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        return pd.read_csv(latest_file)
        
    def train_model(self) -> Tuple[Pipeline, dict]:
        """Train the model and return pipeline and metrics"""
        # Load data
        df = self.load_latest_features()
        
        # Split features and target
        X = df[self.features]
        y = df['is_churned']
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Train
        model_pipeline.fit(X_train, y_train)
        
        # Get metrics
        metrics = {
            'train_score': model_pipeline.score(X_train, y_train),
            'test_score': model_pipeline.score(X_test, y_test)
        }
        
        return model_pipeline, metrics
        
    def save_model(self, model: Pipeline, metrics: dict) -> None:
        """Save model with timestamp and metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = self.model_path / f"model_{timestamp}.joblib"
        metrics_file = self.model_path / f"metrics_{timestamp}.txt"
        
        joblib.dump(model, model_file)
        with open(metrics_file, 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
                
        logger.info(f"Model saved to {model_file}")
        logger.info(f"Metrics: {metrics}")

    def run_training(self) -> None:
        """Execute full training pipeline"""
        try:
            model, metrics = self.train_model()
            self.save_model(model, metrics)
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    trainer = ModelTrainer(
        feature_path="data/features",
        model_path="data/models"
    )
    trainer.run_training()