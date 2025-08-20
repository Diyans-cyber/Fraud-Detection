# fraud_detection.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# -----------------------------
# 2. Basic EDA
# -----------------------------
print(df.info())
print(df.describe())

# Class distribution (Anomaly detection: Fraud cases are very rare)
print("Class distribution:\n", df['Class'].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Class Distribution (0 = Normal, 1 = Fraud)")
plt.show()

# -----------------------------
# 3. Feature Engineering
# -----------------------------
# Scale the 'Amount' column (Time can also be scaled/ignored)
scaler = StandardScaler()
df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])

# Drop original 'Amount' and 'Time' (optional, they don’t help much)
df = df.drop(['Amount','Time'], axis=1)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# -----------------------------
# 5. Handle Imbalance (SMOTE)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_res.value_counts())

# -----------------------------
# 6. Machine Learning Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    print(f"\n---- {name} ----")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred
    
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# -----------------------------
# 7. Real-time Monitoring Simulation
# -----------------------------
print("\n--- Real-time Fraud Detection Simulation ---")
sample = X_test.sample(5, random_state=42)

for i in range(len(sample)):
    transaction = sample.iloc[i].values.reshape(1, -1)
    prediction = models["Random Forest"].predict(transaction)[0]
    if prediction == 1:
        print(f"Transaction {i+1}: 🚨 Fraudulent Detected!")
    else:
        print(f"Transaction {i+1}: ✅ Normal Transaction")
