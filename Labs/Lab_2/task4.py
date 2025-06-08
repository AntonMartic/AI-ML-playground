import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

##### Load the data here #####
df = pd.read_pickle('Task_4_data.pkl')

##### Convert the data into numpy arrays here ########
X = df[['Weight', 'Horsepower', 'Frontal area', 'Gear ratio', 'Tire inflation',
       'Engine size']].values  # Features
y = df[['Fuel efficiency']].values   # Target values

# Add a column of ones to X for the bias term
X_with_bias = np.c_[np.ones(X.shape[0]), X]  # Adds a column of ones at the beginning

num_features = X_with_bias.shape[1]
print(f"Number of features (including bias): {num_features}")

learning_rate = 0.3

def cost_function(X, y, weights):
    predictions = X @ weights
    return np.mean((predictions - y) ** 2)

def cost_function_gd(X, y, weights):
    predictions = X @ weights
    return (2 / len(y)) * X.T @ (predictions - y)

# Gradient descent
def optimizer_5(X, y, init_weights, iterations):
    weights = init_weights.copy()
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        weights -= learning_rate * grad
        costs.append(cost_function(X, y, weights))
    return weights, costs

# Initialize weights with an additional element for the bias term
init_weights = np.zeros((num_features, 1))  # Shape: (num_features + 1, 1)
iterations = 100

# Run the optimizer
final_weights, cost_history = optimizer_5(X_with_bias, y, init_weights, iterations)

# Print the final weights (including bias)
print("Final weights (including bias):")
print(final_weights)

# Plot cost function over iterations
plt.plot(range(1, len(cost_history) + 1), cost_history, linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (Cost)")
plt.title("Gradient Descent Convergence")
plt.grid(True)
plt.show()