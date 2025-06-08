import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

##### Load the data here #####
df = pd.read_pickle('Task_3_data.pkl')



##### Convert the data into numpy arrays here ########
X = df[['Spectral detail', 'Spatial detail', 'Noise segment size',
       'Noise level', 'Contrast', 'Lighting quality']].values  # Features
y = df[['Segmentation quality']].values   # Target values

num_features = X.shape[1]
print(f"Number of features: {num_features}")

########## End of Data preperation ##############

learning_rate = 0.3
def cost_function(X, y, weights):
    predictions = X @ weights
    return np.mean((predictions - y) ** 2)


def cost_function_gd(X, y, weights):
    predictions = X @ weights
    return (2 / len(y)) * X.T @ (predictions - y)

# is this ADAM?
def optimizer_1(X, y, init_weights, iterations):
    weights = init_weights.copy()
    beta1, beta2 = 0.9, 0.99 # 0.2, 0.2 from begining
    epsilon = 1e-8
    m = np.zeros_like(weights)
    v = np.zeros_like(weights)
    costs = []
    for t in range(1, iterations + 1):
        grad = cost_function_gd(X, y, weights)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        costs.append(cost_function(X, y, weights))
    return weights, costs

# is this RMSProp?
def optimizer_2(X, y, init_weights, iterations):
    weights = init_weights.copy()
    epsilon = 1e-8
    decay_rate = 0.9 # 0.1 from begining
    grad_accum = np.zeros_like(weights)
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        grad_accum = decay_rate * grad_accum + (1 - decay_rate) * grad ** 2
        adjusted_lr = learning_rate / (np.sqrt(grad_accum) + epsilon)
        weights -= adjusted_lr * grad
        costs.append(cost_function(X, y, weights))
    return weights, costs

# is this Momentum optimizer?
def optimizer_3(X, y, init_weights, iterations):
    weights = init_weights.copy()
    v = np.zeros_like(weights)  # Velocity term
    momentum = 0.1
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        v = momentum * v - learning_rate * grad
        weights += v
        costs.append(cost_function(X, y, weights))
    return weights, costs

# is this AdaGrad?
def optimizer_4(X, y, init_weights, iterations):
    weights = init_weights.copy()
    epsilon = 1e-8
    grad_accum = np.zeros_like(weights)
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        grad_accum += grad ** 2
        adjusted_lr = learning_rate / (np.sqrt(grad_accum) + epsilon)
        weights -= adjusted_lr * grad
        costs.append(cost_function(X, y, weights))
    return weights, costs

# is this gradiant descent?
def optimizer_5(X, y, init_weights, iterations):
    weights = init_weights.copy()
    costs = []
    for _ in range(iterations):
        grad = cost_function_gd(X, y, weights)
        weights -= learning_rate * grad
        costs.append(cost_function(X, y, weights))
    return weights, costs


##### Run Optimizers and Check Convergence #####
init_weights = np.zeros((num_features, 1)) # Ensure weights have correct shape

iterations = 20

optimizers = {
    "Adam": optimizer_1,
    "RMSProp": optimizer_2,
    "Momentum": optimizer_3,
    "AdaGrad": optimizer_4,
    "SGD": optimizer_5
}

plt.figure(figsize=(10, 6))
for name, optimizer in optimizers.items():
    _, costs = optimizer(X, y, init_weights, iterations)
    plt.plot(costs, label=name)
    #if abs(costs[-1] - costs[0]) < 1e-3:
    #    print(f"{name} converged in less than 20 iterations.")
    #else:
    #    print(f"{name} did NOT converge in 20 iterations.")
    final_cost = costs[-1]  # Get the final cost after 20 iterations
    print(f"{name}: Final cost after {iterations} iterations = {final_cost:.6f}")

plt.xlabel("Iterations")
plt.ylabel("Cost Function")
plt.title("Convergence of Different Optimizers")
plt.legend()
plt.show()






