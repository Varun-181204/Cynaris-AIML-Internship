# Local LLM Comparison

Both models received the same system prompt and user questions. Latency includes local inference time and may vary with model loading.

## Question 1

Explain machine learning to a beginner in exactly three sentences.

### llama3.2:3b

Status: success

Machine learning is a type of artificial intelligence that enables computers to learn and improve their performance on a task without being explicitly programmed for it. It works by training algorithms on large datasets, allowing the computer to identify patterns and make predictions or decisions based on that data. As the algorithm is trained and updated, it can adapt to new information and improve its performance over time, making it a powerful tool for applications such as image recognition, natural language processing, and more.

Latency (seconds): 14.958

### qwen2.5:3b

Status: success

Machine learning is a type of artificial intelligence that allows software applications to become more accurate in predicting outcomes without being explicitly programmed. It does this by using algorithms that can learn from and make predictions or decisions based on data inputs. Essentially, it's about teaching computers to improve their performance on a task through experience, making them better at tasks like image recognition, language translation, and recommendation systems.

Latency (seconds): 26.223

## Question 2

What is the difference between classification and regression? Give one example of each.

### llama3.2:3b

Status: success

**Classification**:
Classification is a supervised learning technique where the goal is to predict a categorical label or class that an instance belongs to. The output is a probability distribution over the classes, and the model aims to maximize the accuracy of the predicted class.

**Example of Classification:**
Credit Risk Assessment: A bank wants to predict whether a customer is likely to default on a loan based on their credit score, income, and other factors. The output would be a binary label (0 for "not default" and 1 for "default").

**Regression**:
Regression is a supervised learning technique where the goal is to predict a continuous value or output variable. The output is a single value that best predicts the target variable.

**Example of Regression:**
Predicting House Prices: A real estate company wants to predict the price of a house based on its features such as number of bedrooms, square footage, and location. The output would be a continuous value (e.g., $500,000) that best predicts the house price.

Latency (seconds): 37.663

### qwen2.5:3b

Status: success

Classification and regression are both types of supervised machine learning techniques used for predictive modeling, but they differ in the type of output they produce and the nature of the problem they solve.

**Classification** involves predicting a discrete output based on input features. The output is a category or class. Examples include:
- **Example of Classification**: Predicting whether an email is spam (spam/not spam). The model will output one of two categories: spam or not spam.

**Regression**, on the other hand, predicts a continuous output based on input features. The output is a real number. Examples include:
- **Example of Regression**: Predicting the price of a house based on features like the number of rooms, location, and square footage. The output will be a continuous value, such as $300,000 or $500,000.

In summary, classification deals with predicting categories, while regression deals with predicting numerical values.

Latency (seconds): 36.211

## Question 3

A model correctly predicts 90 out of 120 test examples. Calculate its accuracy as a percentage.

### llama3.2:3b

Status: success

To calculate the accuracy, divide the number of correct predictions (90) by the total number of test examples (120), and multiply by 100.

Accuracy = (90 / 120) * 100
Accuracy = 0.75 * 100
Accuracy = 75%

The model's accuracy is 75%.

Latency (seconds): 20.634

### qwen2.5:3b

Status: success

To calculate the accuracy of the model as a percentage, we use the formula:

\[ \text{Accuracy} = \left( \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} \right) \times 100 \]

Given that the model correctly predicts 90 out of 120 test examples, we can plug these values into the formula:

\[ \text{Accuracy} = \left( \frac{90}{120} \right) \times 100 = 0.75 \times 100 = 75\% \]

So, the accuracy of the model is 75%.

Latency (seconds): 39.785
