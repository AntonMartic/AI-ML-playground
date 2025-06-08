import numpy as np
import matplotlib.pyplot as plt


def estimate_pi(N):
    x_values = np.random.random(N)
    y_values = np.random.random(N)

    xc = yc = 0.5

    M = np.sum(np.sqrt((x_values - xc)**2 + (y_values - yc)**2) < 0.5)

    return (M/N) * 4

# Testing for different N values
for N in [1000, 5000, 10000, 50000, 100000]:
    print(f"N = {N}, Estimated π = {estimate_pi(N)}")

# N value between 5000-10000 seem to estimate pi with two decimals correctly

# Bonus part
N_values = np.arange(1000, 50000, 100)

pi_estimate = [estimate_pi(N) for N in N_values]

errors = np.abs(np.diff(pi_estimate))

plt.plot(N_values[:-1], errors, marker='o', linestyle='-', color='b')
plt.xlabel("N (Number of Random Points)")
plt.ylabel("Error |π₂ - π₁|")
plt.grid(True)
plt.show()