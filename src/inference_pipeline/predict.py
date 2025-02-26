# src/inference_pipeline/predict.py
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChurnPredictor:
    def __init__(self, model_path: str, feature_path: str, output_path: str):
        self.model_path = Path(model_path)
        self.feature_path = Path(feature_path)
        self.output_path = Path(output_path)
        
    def load_latest_model(self):
        """Load the most recent model"""
        model_files = list(self.model_path.glob("model_*.joblib"))
        if not model_files:
            logger.error(f"No model files found in {self.model_path}")
            raise FileNotFoundError(f"No model files found in {self.model_path}")
        
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading model from {latest_model}")
        return joblib.load(latest_model)
    
        
    def load_latest_features(self):
        """Load the most recent feature set"""
        feature_files = list(self.feature_path.glob("features_*.csv"))
        if not feature_files:
            logger.error(f"No feature files found in {self.feature_path}")
            raise FileNotFoundError(f"No feature files found in {self.feature_path}")
        
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading features from {latest_file}")
        return pd.read_csv(latest_file)

    def generate_predictions(self):
        """Generate and save predictions"""
        try:
            # Ensure output directory exists
            self.output_path.mkdir(parents=True, exist_ok=True)
            
            # Load model and features
            model = self.load_latest_model()
            features = self.load_latest_features()
            
            # Get feature names from training features
            feature_names = [
                'bounce_rate', 'pages_per_session',
                'sessions', 'pageviews', 'dsls',
                'visited_demo_page', 'visited_water_purifier_page',
                'visited_vacuum_cleaner_page',
                'help_me_buy_evt_count', 'phone_clicks_evt_count'
            ]
            
            # Check if all required features are present
            missing_features = [f for f in feature_names if f not in features.columns]
            if missing_features:
                logger.error(f"Missing required features: {missing_features}")
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Generate predictions using only the needed features
            predictions = model.predict(features[feature_names])
            probabilities = model.predict_proba(features[feature_names])
            
            # Create predictions DataFrame
            # Fix the customer_id issue - use customer_id column if it exists, otherwise use index
            if 'customer_id' in features.columns:
                customer_ids = features['customer_id']
            else:
                customer_ids = features.index
            
            results = pd.DataFrame({
                'customer_id': customer_ids,
                'churn_prediction': predictions,
                'churn_probability': probabilities[:, 1]
            })
            
            # Save predictions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_path / f"predictions_{timestamp}.csv"
            results.to_csv(output_file, index=False)
            
            logger.info(f"Predictions saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

if __name__ == "__main__":
    predictor = ChurnPredictor(
        model_path="data/models",
        feature_path="data/features",
        output_path="data/predictions"
    )
    predictor.generate_predictions()