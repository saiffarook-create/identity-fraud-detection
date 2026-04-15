# Presentation Deck Outline

## Slide 1: Title

**Identity Theft Detection Using Machine Learning**

- Course name
- Team member name(s)
- Instructor name
- Submission date

Speaker note:
Introduce the project as a login fraud and account takeover detection system using machine learning.

## Slide 2: Problem Background

- Identity theft and account takeover are common cybersecurity threats
- Stolen credentials can be used to access sensitive user accounts
- Manual rules are useful but not always enough
- Machine learning can detect suspicious behavior patterns automatically

Speaker note:
Explain why this matters in banking, e-commerce, and digital platforms.

## Slide 3: Project Objective

- Build a binary classifier for login fraud detection
- Distinguish legitimate logins from suspicious account takeover attempts
- Compare three ML models
- Produce a reusable prediction pipeline and visual results

Speaker note:
Keep the project goal simple and concrete.

## Slide 4: Dataset

- Source: Kaggle login fraud detection dataset
- 450 cleaned rows
- 25 columns
- Binary target: `label`
- Synthetic but cybersecurity-focused

Speaker note:
Mention that the original CSV had metadata banner rows and had to be cleaned first.

## Slide 5: Important Features

- login attempts in last hour
- failed logins in last 24 hours
- password entropy
- VPN usage
- Tor exit node flag
- device fingerprint match
- geo velocity
- payload anomaly score

Speaker note:
These are realistic indicators of suspicious login behavior.

## Slide 6: Data Preprocessing

- removed duplicates
- excluded identifier columns
- handled mixed numeric and categorical data
- scaled numeric features
- one-hot encoded categorical features
- applied stratified train/validation/test split

Speaker note:
Say that preprocessing was built inside a reusable pipeline to avoid leakage.

## Slide 7: Feature Engineering

- created failed login risk flag
- created rare device-location flag
- extracted hour and day from timestamp
- added night login indicator
- selected top 20 features using `SelectKBest`

Speaker note:
These features are simple, explainable, and beginner-friendly.

## Slide 8: Models Compared

- Dummy Classifier
- Logistic Regression
- Random Forest

Why these models:

- baseline comparison
- interpretability
- low implementation complexity
- good academic coverage

## Slide 9: Evaluation Metrics

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Cross-validation

Speaker note:
Explain why recall matters in fraud detection because missed fraud is costly.

## Slide 10: EDA Visualizations

Insert:

- class distribution figure
- correlation heatmap
- missing values chart

Speaker note:
Briefly explain what the dataset looks like before training.

## Slide 11: Model Comparison Results

Use a table:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 |
| Dummy | 0.00 | 0.00 | 0.00 | 0.50 |

Speaker note:
Point out that the baseline fails while the learned models perform perfectly on this dataset.

## Slide 12: Best Model

- Selected model: Logistic Regression
- Chosen because performance matched the best result
- Easier to explain than Random Forest
- Good fit for a beginner-friendly academic project

Insert:

- feature importance plot

Speaker note:
Highlight the top features: payload anomaly score, password entropy, failed logins.

## Slide 13: Confusion Matrix and ROC Curve

Insert:

- confusion matrix
- ROC curve

Speaker note:
Explain that the model separated both classes perfectly on the test split.

## Slide 14: Sample Predictions / Demo

- Show 2-3 example rows from `sample_predictions.csv`
- Include fraud probability
- Include final prediction
- Mention `prediction_reason`

Suggested examples:

- low-risk SG session predicted as legitimate
- high-risk KP session predicted as fraud
- high-risk scripted request session predicted as fraud

Speaker note:
This is your best demo slide because it makes the system feel real.

## Slide 15: Limitations and Future Work

Limitations:

- synthetic dataset
- likely easier than real-world data
- small dataset size
- no live production integration

Future work:

- use real login telemetry
- add SHAP explainability
- build Streamlit dashboard
- support alert prioritization

## Slide 16: Conclusion

- Built a full ML pipeline for identity theft detection
- Compared three models
- Generated report-ready artifacts
- Achieved strong results on the selected Kaggle dataset
- Demonstrated how ML can support account takeover detection

Speaker note:
End with a clear takeaway: the project is complete, reproducible, and relevant to cybersecurity.

## Slide 17: Q&A

- Thank you
- Questions?

### Likely Questions and Suggested Answers

**Why are the results perfect?**

The dataset is synthetic and likely highly separable, so the patterns are easier than in real-world fraud data.

**Why choose logistic regression over random forest?**

Both performed equally well, but logistic regression is simpler and easier to interpret.

**What are the strongest fraud signals?**

Payload anomaly score, password entropy, and failed login count.

**What would you improve next?**

Test on real-world data, add explainability, and build a simple dashboard.
