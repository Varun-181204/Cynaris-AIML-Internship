# Model Card: W4D1 Binary Classification Model

## 1. Model details
- Model: StandardScaler followed by Logistic Regression
- Task: Binary classification
- Framework: scikit-learn Pipeline
- Random seed: 42
- Original exercise: Cynaris AI/ML Internship, Week 4 Day 1

## 2. Intended use
This model demonstrates reproducible model training, preprocessing, evaluation, cross-validation, and explanation. It is intended for educational experimentation only. It must not be used to make automated high-impact decisions.

## 3. Training data
The model uses a synthetic dataset generated with scikit-learn's make_classification: 1,500 samples, 20 numerical features, 8 informative features, 4 redundant features, and an approximate 70/30 class balance. The dataset contains no real individuals' records.

## 4. Training procedure
The data is split into training and test sets using stratification and random seed 42. StandardScaler and LogisticRegression are combined in a Pipeline. Scaling is fitted on training data only. The original exercise uses max_iter=2000.

## 5. Evaluation results
- Test accuracy: 0.8067
- Test precision: 0.8511
- Test recall: 0.4396
- Test F1-score: 0.5797
- Test ROC-AUC: 0.8393
- Stratified five-fold cross-validation ROC-AUC: 0.8114 ± 0.0289

These values are from the W4D1 evaluation report. The SHAP demonstration refits the same model configuration on a stratified split; its split accuracy is recorded separately in shap_summary_metrics.json.

## 6. Fairness and explainability
The W4D6 exercise separately evaluates gender-related group metrics on the UCI Adult Income dataset and demonstrates AIF360 Reweighing. SHAP is used to explain the W4D1 synthetic model. Adult Income results are not a fairness audit of the W4D1 model.

## 7. Limitations and ethical considerations
Synthetic features do not represent real-world demographic characteristics. Consequently, demographic fairness cannot be established for the W4D1 model. The model has not been validated for hiring, lending, healthcare, policing, or other high-impact use. SHAP describes model behavior but does not prove causation or eliminate bias.

## 8. Monitoring and future work
Future work should use a documented, representative dataset, assess subgroup performance where appropriate protected attributes are available, conduct independent reviews, monitor performance over time, retain human oversight, and provide a way to challenge or correct harmful decisions.
