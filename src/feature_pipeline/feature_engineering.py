import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Tuple, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering pipeline for customer churn prediction.
    Handles data loading, feature creation, and basic validation.
    """
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Define feature sets
        self.engagement_features = [
            'sessions', 'pageviews', 'bounces',
            'sessionDuration', 'dsls'
        ]
        
        self.product_features = [
            'visited_demo_page',
            'visited_water_purifier_page',
            'visited_vacuum_cleaner_page'
        ]
        
        self.support_features = [
            'help_me_buy_evt_count',
            'phone_clicks_evt_count',
            'DemoReqPg_CallClicks_evt_count'
        ]
        
        self.static_features = [
            'region', 'country', 'device', 'sourceMedium'
        ]

    def create_churn_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates churn labels based on defined conditions.
        """
        logger.info("Creating churn labels...")
        
        df['is_churned'] = (
            (df['dsls'] > 30) |  # No activity for 30 days
            ((df['sessions'] == 0) & (df['sessions_hist'] > 0)) |  # Zero sessions with history
            (df['sessions'] < 0.2 * df['sessions_hist'])  # 80% drop in sessions
        ).astype(int)
        
        return df

    def create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates engagement-related features.
        """
        logger.info("Creating engagement features...")
        
        # Create basic ratio features
        df['bounce_rate'] = df['bounces'] / df['sessions']
        df['pages_per_session'] = df['pageviews'] / df['sessions']
        
        # Handle infinities and NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df

    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Basic feature validation checks.
        """
        logger.info("Validating features...")
        
        # Check for missing values
        missing_pct = df.isnull().mean() * 100
        if missing_pct.max() > 10:  # Alert if any feature has >10% missing
            logger.warning(f"High missing values detected: {missing_pct[missing_pct > 10]}")
            
        # Check for infinite values in numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if df[numeric_cols].isin([np.inf, -np.inf]).any().any():
            logger.warning("Infinite values detected in numeric features")
            
        return True

    def run_pipeline(self) -> None:
        """
        Executes the full feature engineering pipeline.
        """
        try:
            # Load data
            logger.info(f"Loading data from {self.input_path}")
            df = pd.read_csv(self.input_path)
            
            # Create features
            df = self.create_churn_label(df)
            df = self.create_engagement_features(df)
            
            # Validate features
            self.validate_features(df)
            
            # Save features with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_path / f"features_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            logger.info(f"Features saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    feature_engineer = FeatureEngineer(
        input_path="data/raw/data_55k_clean.csv",
        output_path="data/features"
    )
    feature_engineer.run_pipeline()