import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

data = pd.DataFrame({
    'interest_similarity': [0.8, 0.2, 0.6, 0.9, 0.3],
    'skill_match': [0.7, 0.1, 0.5, 0.95, 0.4],
    'communication_score': [0.9, 0.3, 0.6, 0.8, 0.2],
    'compatibility': [1, 0, 1, 1, 0]
})

X = data.drop('compatibility', axis=1)
y = data['compatibility']
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

importances = model.feature_importances_
feature_names = poly.get_feature_names_out(X.columns)

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df.head(10))

