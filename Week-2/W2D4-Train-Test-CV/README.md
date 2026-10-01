\# W2D4 — Train/Test Split \& Cross-Validation



\## Objective



Implement a proper train/test split and cross-validation workflow

for a classification problem while preventing data leakage.



\## Implementation



The implementation:



1\. Loads the Iris dataset.

2\. Separates features and target.

3\. Uses an 80/20 stratified train/test split.

4\. Builds a Pipeline containing StandardScaler and LogisticRegression.

5\. Performs 5-fold Stratified Cross-Validation on the training data.

6\. Calculates mean and standard deviation of CV accuracy.

7\. Trains the final pipeline on the complete training set.

8\. Evaluates the final model on the unseen test set.

9\. Saves numerical and visual output evidence.



\## Why Train/Test Split?



The dataset is divided into training and testing data so the final

model can be evaluated on data that was not used during training.



Stratification preserves the class distribution in both subsets.



\## Why Cross-Validation?



5-fold cross-validation provides multiple validation measurements

instead of relying on a single validation split. This gives a better

estimate of model performance and shows variation between folds.



\## Why Use a Pipeline?



StandardScaler is placed inside the Pipeline so scaling is fitted

separately within each cross-validation training fold.



This prevents information from the validation folds from leaking into

the preprocessing step.



\## Results



\- Dataset: 150 samples

\- Features: 4

\- Training samples: 120

\- Testing samples: 30

\- Cross-validation: 5-fold StratifiedKFold

\- Mean CV accuracy: 0.9583

\- CV standard deviation: 0.0264

\- Test accuracy: 0.9333



\### Cross-Validation Scores



| Fold | Accuracy |

|---|---:|

| 1 | 0.9583 |

| 2 | 1.0000 |

| 3 | 0.9583 |

| 4 | 0.9583 |

| 5 | 0.9167 |

| Mean | 0.9583 |



\## Output Evidence



\- `outputs/cross\_validation\_results.csv`

\- `outputs/cross\_validation\_accuracy.png`



\## Viva Questions



\### 1. Explain what you built today and why you made your key design decisions.



I built a train/test and cross-validation workflow using the Iris

dataset. I used an 80/20 stratified split to preserve class

distribution and 5-fold StratifiedKFold cross-validation to obtain

multiple validation measurements. A Pipeline was used so scaling was

performed correctly within each training fold.



\### 2. What was the hardest part? How did you solve it?



The most important challenge was avoiding data leakage during

cross-validation. I solved this by placing StandardScaler and the

classification model inside a Pipeline.



\### 3. If you had one more day, what would you improve?



I would compare multiple classification algorithms and evaluate them

using additional metrics such as precision, recall, F1-score, and

confusion matrix.



\## Self-Review Checklist



\- \[x] Train/test split implemented.

\- \[x] Stratification used.

\- \[x] 5-fold cross-validation implemented.

\- \[x] StandardScaler included in a Pipeline.

\- \[x] Data leakage prevention considered.

\- \[x] Mean CV accuracy calculated.

\- \[x] CV standard deviation calculated.

\- \[x] Test-set evaluation completed.

\- \[x] CSV output generated.

\- \[x] Visualization generated.

\- \[x] Code tested successfully.

\- \[x] Viva questions answered.

\- \[x] Documentation completed.

