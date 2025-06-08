import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Importing random forest classifier
from sklearn.ensemble import RandomForestClassifier

# Imports to calculate the accuracy of both classifiers
from sklearn.metrics import accuracy_score

# 1. Load the data
df = pd.read_pickle('Lab1_Task4_data.pkl')
X = df[['Tissue Texture Score', 'Tissue Density Score']].values
y = df['Diagnosis']

# 2. fitting the model to the training data
svm_clf = SVC(kernel='rbf', C=1000)
svm_clf.fit(X, y)

# Create and fit the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=20)
rf_clf.fit(X, y)

# 3. Transforming the test data to find decision boundary
x_texture_min, x_texture_max = int(min(df['Tissue Texture Score'])), int(max(df['Tissue Texture Score']))
x_density_min, x_density_max = int(min(df['Tissue Density Score'])), int(max(df['Tissue Density Score']))
xx_texture, xx_density = np.meshgrid(np.linspace(x_texture_min, x_texture_max, num=100), np.linspace(x_density_min, x_density_max, num=100))
xx_decision_boundary = np.stack((xx_texture.flatten(), xx_density.flatten()), axis=1)
Z_SVM = svm_clf.predict(xx_decision_boundary)
Z_SVM = Z_SVM.reshape(xx_texture.shape)

# Predict the class labels for each point using the random forest classifier.
Z_RF = rf_clf.predict(xx_decision_boundary)
Z_RF = Z_RF.reshape(xx_texture.shape)

# 5. Visualizing the decision boundary
plt.figure()
plt.contourf(xx_texture, xx_density, Z_SVM, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='rainbow')
plt.title("SVM")
plt.xlabel("Tissue Texture Score")
plt.ylabel("Tissue Density Score")
plt.tight_layout()
plt.show()

# Visualize the decision boundary for the Random Forest classifier
plt.contourf(xx_texture, xx_density, Z_RF, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='rainbow')
plt.title("Random Forest Decision Boundary")
plt.xlabel("Tissue Texture Score")
plt.ylabel("Tissue Density Score")
plt.show()

# Create a figure with two subplots
plt.figure(figsize=(8, 8))

# Subplot 1: SVM Decision Boundary
plt.subplot(2, 1, 1)
plt.contourf(xx_texture, xx_density, Z_SVM, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='rainbow')
plt.title("SVM Decision Boundary")
plt.xlabel("Tissue Texture Score")
plt.ylabel("Tissue Density Score")
plt.axis('equal')

# Subplot 2: Random Forest Decision Boundary
plt.subplot(2, 1, 2)
plt.contourf(xx_texture, xx_density, Z_RF, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8, cmap='rainbow')
plt.title("Random Forest Decision Boundary")
plt.xlabel("Tissue Texture Score")
plt.ylabel("Tissue Density Score")
plt.axis('equal')

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

# Predict on the training data
y_pred_svm = svm_clf.predict(X)
y_pred_rf = rf_clf.predict(X)

# Calculate accuracy
accuracy_svm = accuracy_score(y, y_pred_svm)
accuracy_rf = accuracy_score(y, y_pred_rf)

print(f"SVM Accuracy: {accuracy_svm:.2f}")
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
