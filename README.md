# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: DHIVYA DHARSHINI P
RegisterNumber: 212225220028
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\acer\Downloads\50_Startups.csv")
X=data['R&D Spend'].values
y=data['Profit'].values

X=(X-X.mean())/X.std()
m=0
b=0
learning_rate=0.01
epochs=1000
n=len(X)

for i in range(epochs):
    y_pred=m*X+b

    dm=(-2/n)*np.sum(X*(y-y_pred))
    db=(-2/n)*np.sum(y-y_pred)

    m=m-learning_rate*dm
    b=b-learning_rate*db

print("Slope(m):",m)
print("Intercept(b):",b)

y_pred=m*X+b

plt.scatter(X,y)
plt.plot(X,y_pred)

plt.xlabel("R&D Spend (Normalized)")
plt.ylabel("Profit")
plt.title("Gradient Descent on 50_Startups Dataset")
plt.show()

```

## Output:
<img width="1870" height="819" alt="Screenshot 2026-04-29 154342" src="https://github.com/user-attachments/assets/22c0e1c0-aa11-4b22-8e99-ea0220c0fa9f" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
