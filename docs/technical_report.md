# Identity Theft Detection Using Machine Learning

## Executive Summary

This project presents a beginner-friendly machine learning system for detecting identity theft and account takeover behavior from login-session data. The project was developed as a university cybersecurity machine learning assignment and focuses on building a complete, reproducible pipeline rather than an overly complex production system.

The implemented system uses a synthetic but realistic login fraud dataset downloaded from Kaggle: `karanamshrivasta/login-fraud-detection-dataset`. The dataset contains 450 login-session records with 25 columns describing session behavior, device indicators, geographic risk, failed logins, password quality, and anomaly-related measurements. The task is formulated as a binary classification problem where `0` represents legitimate behavior and `1` represents fraudulent or suspicious login activity.

The final solution includes data loading, preprocessing, feature engineering, model training, evaluation, artifact saving, and a command-line inference pipeline. Three machine learning models were compared: a dummy baseline classifier, logistic regression, and random forest. The best-performing model was logistic regression.

On the selected test split, the final model achieved:

- Precision: `1.00`
- Recall: `1.00`
- F1-score: `1.00`
- ROC-AUC: `1.00`

Although these results are excellent, the report also notes an important limitation: the dataset is synthetic and appears highly separable. This means the performance may overestimate what would be possible on noisy real-world financial or authentication logs. Even with that limitation, the project successfully demonstrates how machine learning can be used to detect fraud-related behavior patterns such as high failed login counts, high anomaly scores, weak passwords, suspicious geolocation changes, and risky access infrastructure such as VPN or Tor usage.

The project is suitable for academic demonstration because it satisfies the full end-to-end lifecycle of a cybersecurity ML system: data understanding, preprocessing, feature creation, model comparison, evaluation, and sample predictions.

## 1. Introduction

Identity theft and account takeover are major cybersecurity problems that affect banks, e-commerce platforms, cloud services, and consumer applications. Attackers often use stolen credentials, password stuffing, session hijacking, device spoofing, or automated bots to gain unauthorized access to user accounts. Traditional rule-based systems can detect some suspicious behavior, but they often struggle when attackers change tactics or combine several weak signals into one successful attack.

Machine learning offers a useful alternative because it can learn patterns from past suspicious behavior and classify future sessions as likely fraudulent or legitimate. In this project, machine learning is used to classify login sessions based on behavioral and technical signals such as failed logins, device mismatch, anomaly scores, login timing, VPN use, Tor exit usage, and geographic velocity.

The goal of this project is to build a beginner-friendly identity theft detection system that:

- takes login-session records as input
- predicts whether the behavior is legitimate or fraudulent
- compares multiple machine learning models
- produces report-ready metrics and visualizations
- supports reusable inference on unseen records

## 2. Problem Statement

The problem addressed in this project is binary classification of login sessions for fraud detection.

Given a set of session features, the model predicts:

- `0`: legitimate session
- `1`: fraudulent session / suspicious account takeover attempt

This problem is relevant to cybersecurity because login fraud is often one of the earliest signals of identity misuse. Detecting it quickly can reduce financial loss, prevent data theft, and protect user accounts from compromise.

### Success Criteria

The project is considered successful if it:

- builds a working training and prediction pipeline
- compares at least three models
- reports precision, recall, F1-score, ROC-AUC, and confusion matrix
- generates report-ready figures
- saves a usable trained model
- supports at least 10 sample predictions

## 3. Dataset Description

### 3.1 Dataset Source

The dataset used in this project was downloaded from Kaggle:

- Dataset: `Login Fraud Detection Dataset`
- Author: `Karanam Shrivasta`
- Kaggle reference: `karanamshrivasta/login-fraud-detection-dataset`

The dataset is synthetic, which makes it appropriate for safe academic experimentation while still preserving realistic fraud-related patterns.

### 3.2 Dataset Size and Structure

After cleaning, the dataset contains:

- Rows: `450`
- Columns: `25`

Important columns include:

- `timestamp`
- `dst_port`
- `country_code`
- `user_agent`
- `login_attempts_last_1h`
- `failed_logins_last_24h`
- `session_duration_sec`
- `bytes_sent`
- `bytes_received`
- `num_requests_per_session`
- `password_entropy`
- `account_age_days`
- `is_vpn`
- `is_tor_exit_node`
- `device_fingerprint_match`
- `geo_velocity_kmph`
- `new_device`
- `captcha_triggered`
- `login_hour`
- `weekend_login`
- `payload_anomaly_score`
- `label`

### 3.3 Target Variable

The target column is `label`, where:

- `0` means normal login behavior
- `1` means suspicious/fraudulent login behavior

### 3.4 Data Issues Found

The downloaded CSV originally contained a metadata banner at the top, beginning with comment lines such as `# DATASET` and `# AUTHOR`. This caused the initial CSV parsing to treat the banner as actual data. To fix this, the file was re-read while skipping comment lines, then exported as a clean CSV for training.

This preprocessing step was important because machine learning pipelines require a valid structured header row.

## 4. Data Preprocessing

The project includes a reproducible preprocessing pipeline implemented in Python using pandas and scikit-learn.

### 4.1 Data Loading

The raw CSV is loaded through a helper function that:

- validates that the file exists
- ensures it is a CSV
- detects malformed Kaggle banner rows when necessary

### 4.2 Cleaning Steps

The following cleaning steps were applied:

- remove duplicate rows
- skip Kaggle metadata banner lines
- preserve the target column separately
- remove identifier columns that should not be used for learning

The following columns were treated as identifier-like fields and excluded from modeling:

- `row_id`
- `session_id`
- `src_ip`

### 4.3 Train / Validation / Test Split

The dataset was split using stratified sampling to preserve class balance:

- Train: `70%`
- Validation: `15%`
- Test: `15%`

Stratified splitting is important in fraud detection because positive and negative classes can be unevenly distributed.

### 4.4 Encoding and Scaling

Two preprocessing branches were used:

- Numeric columns:
  - median imputation
  - standard scaling
- Categorical columns:
  - most-frequent imputation
  - one-hot encoding

This design keeps the pipeline simple while still supporting mixed data types.

### 4.5 Feature Selection

After preprocessing, `SelectKBest` with ANOVA F-score was applied to retain the top 20 features. This helps reduce dimensionality and makes the final model easier to explain.

## 5. Feature Engineering

To improve model usefulness while staying beginner-friendly, only a few explainable engineered features were created.

Added features include:

- `failed_logins_last_24h_flag`
  - indicates whether failed logins exceed a suspicious threshold
- `rare_device_location_flag`
  - identifies uncommon user-agent and location combinations
- `event_hour`
  - extracts the hour from the timestamp
- `event_dayofweek`
  - extracts the weekday from the timestamp
- `is_night_event`
  - flags late-night or early-morning access patterns

These features reflect real-world fraud intuition. For example:

- repeated failed logins suggest password attacks
- rare device-location combinations may indicate account takeover
- night-time access may be suspicious depending on the platform and user population

## 6. Methodology

### 6.1 Models Compared

Three models were trained and compared:

1. `DummyClassifier`
   - baseline model predicting the majority class
2. `LogisticRegression`
   - simple, interpretable classifier with class balancing
3. `RandomForestClassifier`
   - tree-based ensemble model with class balancing

### 6.2 Why These Models

These models were chosen because they provide a good balance between:

- beginner friendliness
- interpretability
- strong baseline performance
- low implementation complexity

The dummy classifier establishes a weak reference point. Logistic regression provides a classic linear baseline that is easy to explain. Random forest captures more complex nonlinear interactions.

### 6.3 Training Pipeline

Each model was trained inside a scikit-learn pipeline containing:

- preprocessing
- feature selection
- estimator

This avoids data leakage and makes the pipeline reusable during inference.

### 6.4 Cross-Validation

Five-fold stratified cross-validation was used on the training set for model comparison. This improves confidence in the evaluation by checking performance stability across multiple splits.

## 7. Evaluation Metrics

Because fraud detection is a classification problem where false negatives are costly, multiple evaluation metrics were used:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Cross-validation mean and standard deviation

These metrics are more appropriate than accuracy alone, especially for fraud detection tasks.

## 8. Results

### 8.1 Model Comparison

The final leaderboard was:

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 |
| Dummy Classifier | 0.00 | 0.00 | 0.00 | 0.50 |

### 8.2 Best Model

The selected final model was `LogisticRegression`.

It was chosen because:

- it matched the best performance
- it is easier to explain than random forest
- it fits a beginner-friendly academic project well

### 8.3 Test Performance

Test set results for the final model:

- Precision: `1.00`
- Recall: `1.00`
- F1-score: `1.00`
- ROC-AUC: `1.00`

Per-class support:

- Legitimate class support: `41`
- Fraud class support: `27`

### 8.4 Interpretation of Results

The perfect results suggest the dataset is highly separable. This may happen because the data is synthetic and the fraud patterns are very strong. The top predictive signals identified by the model were:

- `payload_anomaly_score`
- `password_entropy`
- `failed_logins_last_24h`

This is consistent with cybersecurity intuition:

- highly anomalous payloads are suspicious
- weak password behavior can indicate compromise
- repeated failed logins often occur before successful account takeover

## 9. Visualizations

The project generated the following figures:

1. Class distribution
2. Missing values chart
3. Correlation heatmap
4. Numeric feature distribution 1
5. Numeric feature distribution 2
6. Numeric feature distribution 3
7. Model comparison bar chart
8. Confusion matrix
9. ROC curve
10. Feature importance plot

These figures are saved in the `artifacts/figures/` folder and can be inserted directly into the technical report or presentation.

## 10. Sample Predictions

The project also generated sample predictions for unseen records. Examples include:

- A session from Singapore with low failed logins, high password entropy, and low anomaly score was predicted as legitimate with a fraud probability close to `0.001`.
- A suspicious login from North Korea using `libwww-perl`, 25 failed logins, VPN/Tor signals, and a payload anomaly score near `0.993` was predicted as fraud with probability `0.9999`.
- A suspicious login using `python-requests`, 27 failed logins, new device behavior, and extreme anomaly signals was also predicted as fraud with high confidence.

These examples show that the inference pipeline can distinguish between normal and suspicious behavior in a way that is interpretable and useful for demonstration.

## 11. System Artifacts

The project saves the following outputs:

- trained model file
- leaderboard CSV
- metrics JSON
- metadata JSON
- sample predictions CSV
- processed train/validation/test splits
- figures for EDA and evaluation

This makes the workflow reproducible and suitable for grading.

## 12. Limitations

This project has several limitations:

1. The dataset is synthetic rather than real production data.
2. Perfect model performance likely reflects strong signal separation rather than true real-world robustness.
3. The dataset is relatively small compared to real authentication systems.
4. The project does not include live streaming or real-time monitoring infrastructure.
5. Explainability is based on feature importance rather than SHAP or LIME.

These limitations should be acknowledged clearly in the final submission.

## 13. Future Work

The system could be improved in the future by:

- testing on a real-world fraud or login telemetry dataset
- adding threshold tuning for better precision-recall tradeoffs
- incorporating SHAP explanations
- building a Streamlit dashboard
- adding alert severity categories
- using anomaly detection for unseen attack types
- combining supervised learning with rule-based detection

## 14. Conclusion

This project successfully implemented a complete machine learning pipeline for identity theft and login fraud detection. It covered the full workflow from raw data handling to evaluation and inference. The final model performed very well on the selected Kaggle dataset and demonstrated that cybersecurity-related behavioral signals can be used effectively in a supervised learning setting.

For a university-level project, this work is strong because it combines:

- a relevant cybersecurity problem
- a complete and reproducible pipeline
- multiple model comparisons
- report-ready visual outputs
- practical, interpretable predictions

Even though the dataset is synthetic, the project still offers useful insight into how ML systems can support account takeover detection and fraud prevention.

## References

Use 5-6 recent academic papers in IEEE format in the final version. Suggested literature topics:

- account takeover detection
- authentication anomaly detection
- fraud detection with behavioral analytics
- cybersecurity ML interpretability
- login anomaly detection
- supervised learning in intrusion/fraud detection
