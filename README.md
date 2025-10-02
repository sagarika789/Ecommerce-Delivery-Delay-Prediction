# Ecommerce-Delivery-Delay-Prediction
# Two-Stage Delay Prediction for E-commerce Logistics
**Final Year B.Tech Project | Applied Machine Learning & Interpretability**

This project introduces a robust, cascaded ML framework to enhance logistical efficiency by first classifying the likelihood of a delivery delay (Stage 1) and then accurately predicting the duration of that delay (Stage 2).

# Key Contributions & Technology
This solution uses advanced ML techniques to address the high variability and complexity of Indian metropolitan logistics data.

**Stage 1**: Delay Classification

**Model**: LightGBM Classifier, optimized using Optuna for hyperparameter tuning.

**Purpose**: Filters on-time vs. delayed deliveries with high precision.


**Stage 2**: Duration Regression

**Model**: Deep Learning Multi-Layer Perceptron (MLP) Regressor.

**Purpose**: Estimates delay duration (in minutes) for deliveries flagged as late.

# Performance & Impact
The cascaded approach significantly minimized prediction error and provided strong classification results:

| **Model Stage**            | **Primary Goal**              | **Key Performance Metric**       | **Value** |
|-----------------------------|-------------------------------|----------------------------------|-----------|
| **Stage 1 (Classification)** | Identify if delay occurs      | Precision (on predicting delays) | **82%**   |
| **Stage 2 (Regression)**    | Predict duration of delay     | Mean Absolute Error (MAE)        | **1.59 min** |
|                             |                               | Coefficient of Determination (R²)| **99.7%** |


# Interpretability: SHAP Analysis
Feature importance was analyzed using SHAP (SHapley Additive exPlanations) to provide transparency for business decisions. The top contributing factors driving the model's delay prediction are:

* **Geospatial Distance (34.2%)**

* **Traffic Jams (28.1%)**

* **Weekend Orders / Temporal Patterns (19.7%)**

(See the results/shap_plot.png image for the visual feature importance breakdown.)

# Tech Stack
**Languages**: Python

**Libraries**: LightGBM, Scikit-learn, Optuna, SHAP, TensorFlow/Keras (for MLP implementation)

**Engineering**: Feature Engineering (Haversine Distance, Time-based features), Git.

# Dataset & Setup
**Dataset Note**
This project uses the Amazon Delivery Dataset (approx. 6 MB). To run the code, you must manually download the single CSV file and place it in the project root directory.

**Source URL**: https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset

**Required Filename**: amazon_delivery.csv

**Local Installation**:

* Clone the repository:
git clone [https://github.com/sagarika789/delivery-delay-prediction.git](https://github.com/sagarika789/delivery-delay-prediction.git)
cd delivery-delay-prediction

* Install dependencies:
pip install -r requirements.txt

* Execute the full pipeline:
#Executes the end-to-end ML pipeline: preprocessing, modeling, and evaluation.
python src/train_model.py



# Next Steps
* Extend to include the increase in cost estimation due to delay.

* Integrate real-time traffic API data to improve predictive accuracy further.
**
