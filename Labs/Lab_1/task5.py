from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Features (X) and target (y)
X = [
    [2, 0, 0, 0], [0, 0, 1, 1], [2, 1, 0, 2], [0, 0, 0, 0], [2, 0, 0, 1],
    [1, 0, 1, 0], [2, 1, 0, 2], [1, 0, 0, 1], [2, 0, 0, 1], [0, 0, 1, 0],
    [2, 2, 0, 2], [1, 0, 0, 0], [2, 2, 0, 2], [1, 0, 0, 1], [0, 0, 1, 0],
    [2, 1, 1, 1], [1, 0, 0, 1], [2, 2, 0, 2], [1, 0, 1, 0], [2, 1, 0, 1]
]
y = [1, 0, 1, 0, 2, 0, 1, 1, 2, 0, 1, 0, 1, 1, 0, 2, 1, 1, 0, 1]

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X, y)

# Encode the new customer's data
new_customer = {
    "Subscription Period": "<1 year",  # 0
    "Monthly Usage": "High",           # 2
    "Support Calls": "No",             # 0
    "Price Tier": "Standard"           # 1
}
new_customer_encoded = [0, 2, 0, 1]

# Predict the outcome
prediction = clf.predict([new_customer_encoded])
print(f"Predicted Outcome: {prediction[0]}")

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=["Subscription Period", "Monthly Usage", "Support Calls", "Price Tier"],
          class_names=["Canceled", "Active", "Upgraded"])
plt.show()

# Extract information gain from the tree
n_nodes = clf.tree_.node_count
feature = clf.tree_.feature
threshold = clf.tree_.threshold
impurity = clf.tree_.impurity

# Print information gain for each split
for i in range(n_nodes):
    if feature[i] != -2:  # Skip leaf nodes
        print(f"Node {i}: Feature {feature[i]} (Threshold: {threshold[i]}), Impurity: {impurity[i]}")