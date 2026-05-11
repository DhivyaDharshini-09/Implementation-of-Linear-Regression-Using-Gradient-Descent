# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the Cost function.
3.Perform iterations of gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: DHIVYA DHARSHINI P
RegisterNumber: 212225220028
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\acer\Downloads\50_Startups.csv")

# Let's assume 'R&D Spend' predicts 'Profit'
X = df['R&D Spend'].values
y = df['Profit'].values

# Normalize data (important for gradient descent)
X = (X - X.mean()) / X.std()

# Add bias term (x0 = 1)
m = len(X)
X = np.c_[np.ones(m), X]

# Initialize parameters
theta = np.zeros(2)

# Hyperparameters
alpha = 0.01   # learning rate
iterations = 1000

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    return (1/(2*m)) * np.sum((predictions - y) ** 2)

# Gradient Descent
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y

        gradients = (1/m) * X.T.dot(errors)
        theta = theta - alpha * gradients

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Train model
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

print("Final Parameters (theta):", theta)

# Plot cost function
plt.plot(cost_history)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Reduction over Iterations")
plt.show()

# Predictions
predictions = X.dot(theta)

# Plot regression line
plt.scatter(X[:,1], y, color='blue')
plt.plot(X[:,1], predictions, color='red')
plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.show()

```

## Output:
<img width="827" height="612" alt="Screenshot 2026-05-11 151736" src="https://github.com/user-attachments/assets/9eee68a9-8881-4681-a4dd-7f9413e071c9" />
<img width="893" height="585" alt="Screenshot 2026-05-11 151743" src="https://github.com/user-attachments/assets/19064307-72bb-45ac-89dd-fec524c26cd9" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
