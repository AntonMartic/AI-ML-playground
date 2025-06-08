import numpy as np
import pandas as pd

# Load the data
data = np.load('credit_score_fairness_data.npy')  # Replace with the actual file name

print("Data shape:", data.shape)

print(type(data))
print(data)

def confusion_matrix(y_true, y_pred):
    """Calculate the confusion matrix."""
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, FP, TN, FN

def calculate_rates(TP, FP, TN, FN):
    """Calculate TPR, FPR, and FNR."""
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (FN + TP)
    return TPR, FPR, FNR


# Split data by protected attribute
group_0 = data[data[:, 0] == 0]  # Group with protected attribute = 0
group_1 = data[data[:, 0] == 1]  # Group with protected attribute = 1

# Calculate confusion matrices for each group
TP_0, FP_0, TN_0, FN_0 = confusion_matrix(group_0[:, 1], group_0[:, 2])
TP_1, FP_1, TN_1, FN_1 = confusion_matrix(group_1[:, 1], group_1[:, 2])

# Calculate rates for each group
TPR_0, FPR_0, FNR_0 = calculate_rates(TP_0, FP_0, TN_0, FN_0)
TPR_1, FPR_1, FNR_1 = calculate_rates(TP_1, FP_1, TN_1, FN_1)

# Print results
print("Group 0 (Protected attribute = 0):")
print(f"TPR: {TPR_0:.4f}, FPR: {FPR_0:.4f}, FNR: {FNR_0:.4f}")
print("Group 1 (Protected attribute = 1):")
print(f"TPR: {TPR_1:.4f}, FPR: {FPR_1:.4f}, FNR: {FNR_1:.4f}")