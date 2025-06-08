import pandas as pd
import numpy as np

from interpret import set_visualize_provider
from interpret.provider import InlineProvider
set_visualize_provider(InlineProvider())

from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from interpret import show
import matplotlib.pyplot as plt

# Load the data
df = pd.read_json('UCI Heart Disease Prediction.json')

print(df.columns)

# Separate features (X) and target (y)
X = df.drop(columns=['target'])  # Features (all columns except 'target')
y = df['target']  # Target (last column)

print(X.columns)

seed = 42
np.random.seed(seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

# Initialize and train the EBM model
ebm = ExplainableBoostingClassifier()
ebm.fit(X_train, y_train)

# Visualize global explanation
#show(ebm.explain_global())

y_values = ebm.explain_global().data()['scores']
a = len(X.columns)
y_values = y_values[:a]

plt.figure()
plt.bar(X.columns,y_values)
plt.xticks(rotation=45)
plt.show()

#show(ebm.explain_local(X_test[:5], y_test[:5]), 0)

y_values_local = ebm.explain_local(X_train.iloc[[200]], y_train.iloc[[200]]).data(0)['scores']
y_values_local = y_values_local[:a]
print(y_values_local)

plt.figure()
plt.bar(X.columns,y_values_local)
plt.xticks(rotation=45)
plt.show()