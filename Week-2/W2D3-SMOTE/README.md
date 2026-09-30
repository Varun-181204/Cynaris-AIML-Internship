\# W2D3 — Handling Imbalanced Data with SMOTE



\## Objective



Handle class imbalance in a binary classification dataset using

SMOTE (Synthetic Minority Over-sampling Technique).



\## Implementation



The implementation:



1\. Creates an intentionally imbalanced classification dataset.

2\. Splits the data into training and testing sets using stratification.

3\. Scales the training features using StandardScaler.

4\. Applies SMOTE only to the training data.

5\. Compares class distributions before and after SMOTE.

6\. Saves numerical and visual output evidence.



\## Why SMOTE?



SMOTE creates synthetic samples for the minority class instead of

simply duplicating existing minority samples.



This helps create a more balanced training dataset and can improve

a model's ability to learn patterns from the minority class.



\## Important Design Decision



SMOTE is applied only after the train-test split and only to the

training data.



Applying SMOTE before splitting could cause synthetic information

derived from training samples to appear in the test set, resulting

in data leakage.



\## Results



| Class | Before SMOTE | After SMOTE |

|------:|-------------:|------------:|

| 0     | 718          | 718         |

| 1     | 82           | 718         |



The minority class was balanced with the majority class.



\## Output Evidence



\- `outputs/class\_distribution.csv`

\- `outputs/smote\_class\_distribution.png`



\## Viva Questions



\### 1. Explain what you built today and why you made your key design decisions.



I created an imbalanced binary classification dataset and used SMOTE

to balance the minority class. I split the data before applying SMOTE

to prevent data leakage. SMOTE was applied only to the training data,

while the test data remained unchanged for evaluation.



\### 2. What was the hardest part? How did you solve it?



The important part was applying SMOTE at the correct stage of the

machine learning workflow. I solved this by performing the

train-test split first and applying SMOTE only to the training data.



\### 3. If you had one more day, what would you improve?



I would train a classification model before and after SMOTE and

compare metrics such as precision, recall, F1-score, and confusion

matrix to measure the effect of balancing the dataset.



\## Technologies



\- Python

\- Pandas

\- Scikit-learn

\- imbalanced-learn

\- Matplotlib

