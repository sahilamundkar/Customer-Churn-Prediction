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
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        return joblib.load(latest_model)
        
    def load_latest_features(self):
        """Load the most recent feature set"""
        feature_files = list(self.feature_path.glob("features_*.csv"))
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        return pd.read_csv(latest_file)

    def generate_predictions(self):
        """Generate and save predictions"""
        try:
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
            
            # Generate predictions using only the needed features
            predictions = model.predict(features[feature_names])
            probabilities = model.predict_proba(features[feature_names])
            
            # Create predictions DataFrame
            results = pd.DataFrame({
                'customer_id': features.index,
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