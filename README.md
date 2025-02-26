# Customer Churn Prediction System

![GitHub Actions Workflow Status](https://github.com/sahilamundkar/Customer-Churn-Prediction/actions/workflows/model_training.yml/badge.svg)
![GitHub Actions Workflow Status](https://github.com/sahilamundkar/Customer-Churn-Prediction/actions/workflows/batch_inference.yml/badge.svg)

A production-ready machine learning system that predicts customer churn using automated MLOps pipelines.

## Business Value

Customer churn (when customers stop using a product or service) is a critical business challenge that directly impacts revenue and growth. This system:

- **Reduces Revenue Loss**: Identifies at-risk customers before they churn, enabling targeted retention efforts
- **Optimizes Marketing Spend**: Focuses retention resources on customers most likely to churn
- **Improves Customer Experience**: Helps identify pain points in the customer journey
- **Drives Data-Driven Decisions**: Provides actionable insights on customer behavior patterns

For a typical business, reducing churn by just 5% can increase profits by 25-95% (according to research by Bain & Company).

## System Architecture

![MLOps Architecture](https://raw.githubusercontent.com/sahilamundkar/Customer-Churn-Prediction/main/docs/architecture_diagram.png)

The system implements a modern MLOps architecture with three main pipelines:

1. **Feature Pipeline**: Extracts and processes customer data daily
2. **Model Training Pipeline**: Trains and evaluates churn prediction models weekly
3. **Batch Inference Pipeline**: Generates daily churn predictions for business teams

All pipelines are fully automated using GitHub Actions, ensuring reliable and consistent execution.

## Key Technical Features

- **Automated End-to-End ML Workflow**: From data processing to prediction delivery
- **CI/CD Integration**: Automated testing and deployment via GitHub Actions
- **Model Versioning**: Tracking of model artifacts and performance metrics
- **Robust Error Handling**: Comprehensive logging and failure recovery
- **Scalable Design**: Architecture supports growing data volumes

## Pipeline Components

### Feature Pipeline

The feature engineering pipeline extracts raw customer data and transforms it into ML-ready features:

- Runs daily at midnight UTC
- Processes customer interaction data, website analytics, and purchase history
- Creates engagement metrics and behavioral indicators
- Outputs timestamped feature sets for model training

Key features include:
- Engagement metrics (bounce rate, pages per session)
- Session data (frequency, pageviews)
- Product interest indicators (page visits, demo requests)
- Customer support interactions

### Model Training Pipeline

The model training pipeline builds and evaluates churn prediction models:

- Runs weekly on Sundays
- Uses Random Forest classification algorithm
- Performs automated feature selection and hyperparameter tuning
- Evaluates model performance with precision, recall, and F1 metrics
- Stores model artifacts and performance metrics

Current model performance:
- Training accuracy: 83.4%
- Test accuracy: 78.3%
- F1 score: 0.76

### Batch Inference Pipeline

The prediction pipeline applies the latest model to generate churn predictions:

- Runs daily after feature generation
- Loads the most recent model and feature set
- Generates churn probability scores for each customer
- Outputs predictions for consumption by business teams

## Example Outputs

### Feature Set Sample

customer_id,bounce_rate,pages_per_session,sessions,pageviews,dsls,...
1001,0.23,4.5,12,54,3.2,...
1002,0.45,2.1,3,6,0.8,...

### Prediction Output Sample

customer_id,churn_prediction,churn_probability
1001,0,0.23
1002,1,0.87




## Getting Started

### Prerequisites

- Python 3.9+
- Git

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/sahilamundkar/Customer-Churn-Prediction.git
   cd Customer-Churn-Prediction
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up data directories:
   ```
   mkdir -p data/raw data/features data/models data/predictions
   ```

### Running the Pipelines Locally

1. Feature Engineering:
   ```
   python src/feature_pipeline/feature_engineering.py
   ```

2. Model Training:
   ```
   python src/model_pipeline/train.py
   ```

3. Generating Predictions:
   ```
   python src/inference_pipeline/predict.py
   ```

## Deployment

The system is designed to run as GitHub Actions workflows, but can also be deployed to:

- AWS SageMaker
- Azure ML
- Google Cloud AI Platform
- Kubernetes with Kubeflow

## Future Enhancements

- Real-time prediction API using FastAPI
- A/B testing framework for model deployment
- Advanced feature engineering with deep learning
- Explainable AI components for prediction interpretation
- Data drift detection and monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
