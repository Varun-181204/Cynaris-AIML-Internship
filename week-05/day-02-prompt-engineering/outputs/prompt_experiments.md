# Prompt Engineering Experiments

Model: `llama3.2:3b`

Five techniques were tested. Review the actual responses before drawing conclusions about quality.

## Role prompting

Status: success

System prompt: You are a patient machine-learning instructor for beginners.

User prompt: Explain overfitting in machine learning.

Response: Welcome to our machine learning journey! Today, we're going to discuss a crucial concept in machine learning: overfitting.

**What is Overfitting?**

Overfitting is a phenomenon where a machine learning model becomes too specialized to the training data and fails to generalize well to new, unseen data. In other words, the model becomes too good at fitting the noise and patterns in the training data, but not well enough to recognize patterns in new data.

**Why Does Overfitting Happen?**

There are several reasons why overfitting occurs:

1. **Model complexity**: When a model is too complex, it can fit the noise in the training data, but not the underlying patterns.
2. **Insufficient training data**: If the training data is too small, the model may not have enough information to learn from, leading to overfitting.
3. **Overparameterization**: When a model has too many parameters, it can fit the training data too closely, but not generalize well to new data.
4. **Lack of regularization**: Regularization techniques, such as L1 or L2 regularization, can help prevent overfitting by adding a penalty term to the loss function.

**Symptoms of Overfitting**

Overfitting can manifest in several ways:

1. **High training accuracy**: The model performs well on the training data, but poorly on new data.
2. **Low test accuracy**: The model's performance on the test data is significantly lower than its performance on the training data.
3. **Large difference between training and test accuracy**: The difference between the training and test accuracy is large, indicating that the model is overfitting.

**How to Prevent Overfitting**

To prevent overfitting, you can try the following:

1. **Regularization techniques**: Use regularization techniques, such as L1 or L2 regularization, to add a penalty term to the loss function.
2. **Data augmentation**: Increase the size of the training data by applying transformations to the existing data.
3. **Early stopping**: Stop training the model when the test accuracy starts to decrease.
4. **Model simplification**: Simplify the model by reducing its complexity or using a smaller number of parameters.
5. **Cross-validation**: Use cross-validation to evaluate the model's performance on unseen data.

**Example in Python**

Here's an example of overfitting in Python using scikit-learn:
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate some random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 3 * X + np.random.randn(100, 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse}")

# Now, let's see what happens when we add more parameters to the model
X_train_new = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
y_train_new = np.hstack((y_train, np.ones((y_train.shape[0], 1))))

model_new = LinearRegression()
model_new.fit(X_train_new, y_train_new)

# Evaluate the new model on the test data
y_pred_new = model_new.predict(X_test)
mse_new = mean_squared_error(y_test, y_pred_new)
print(f"Test MSE (new model): {mse_new}")
```
In this example, we train a linear regression model on the training data and evaluate its performance on the test data. We then add more parameters to the model and train it again on the new data. The new model performs much worse on the test data, indicating overfitting.

I hope this explanation and example help you understand overfitting in machine learning! Do you have any questions or would you like to explore more topics?

Latency (seconds): 107.424

## Output constraints

Status: success

System prompt: You are a concise technical writer. Follow all output constraints.

User prompt: Explain supervised learning in exactly three bullet points. Each bullet must contain at most 15 words.

Response: Here are three bullet points explaining supervised learning:

• Supervised learning involves training models on labeled datasets to learn patterns and relationships.
• The model learns to predict outputs based on input data and corresponding labels.
• The goal is to minimize error and improve accuracy on unseen, unlabeled data.

Latency (seconds): 7.41

## Few-shot prompting

Status: success

System prompt: Classify the sentiment of text as Positive, Negative, or Neutral. Return only the label.

User prompt: Text: "I love this useful application."
Sentiment: Positive
Text: "The service was terrible."
Sentiment: Negative
Text: "The package arrived today."
Sentiment: Neutral
Text: "The support team solved my problem quickly."
Sentiment:

Response: Positive

Latency (seconds): 2.279

## Structured reasoning

Status: success

System prompt: You are a careful problem-solving tutor. Show a short, clear calculation and then state the answer.

User prompt: A dataset has 240 records. An 80/20 split is used. How many records are in the training and test sets?

Response: To find the number of records in the training and test sets, we need to split the dataset into two parts using the 80/20 ratio.

80% of 240 records = 0.8 x 240 = 192 records
20% of 240 records = 0.2 x 240 = 48 records

So, the training set has 192 records and the test set has 48 records.

Latency (seconds): 10.332

## Audience adaptation

Status: success

System prompt: Explain technical concepts using simple language for a first-year engineering student. Use one practical analogy.

User prompt: What is an API, and why do applications use APIs?

Response: Imagine you're at a restaurant, and you want to order food. You can't just walk into the kitchen and start making your own food, right? That's because the kitchen staff has a system to prepare food, and you need to follow their rules to get what you want.

An API (Application Programming Interface) is like the kitchen staff's system. It's a set of rules and instructions that allows different applications (like your phone or computer) to communicate with each other.

When you use an API, you're essentially saying, "Hey, I want to get some data from your system, and I want to follow your rules to get it." The API then says, "Okay, here's the data you want, but you need to follow these steps to get it."

Applications use APIs for several reasons:

1. **Separation of concerns**: APIs help keep different applications separate and focused on their own tasks. Just like how the kitchen staff prepares food, the API prepares the data.
2. **Reusability**: APIs can be used by multiple applications, making it easier to reuse code and reduce development time.
3. **Scalability**: APIs can handle a large number of requests, making it easier to scale applications and handle increased traffic.
4. **Security**: APIs provide a secure way to exchange data between applications, protecting sensitive information.

In summary, APIs are like the kitchen staff's system that allows applications to communicate with each other, follow rules, and get the data they need to function properly.

Latency (seconds): 32.707
