import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve
)

from sklearn.preprocessing import LabelEncoder
from google.colab import files

import pickle   # ✅ added for saving model

# 📂 Upload CSV
uploaded = files.upload()
file = list(uploaded.keys())[0]
df = pd.read_csv(file)

print("✅ Dataset Loaded Successfully")
print(df.head())
print(df.info())

# ✅ Detect Leak / Status / binary target
possible_targets = [c for c in df.columns if "leak" in c.lower() or "status" in c.lower()]
if possible_targets:
    target = possible_targets[0]
else:
    binary_cols = [c for c in df.columns if df[c].dropna().isin([0,1]).all()]
    if not binary_cols:
        raise ValueError("❌ No leak/status column found! Rename column to 'Leak'")
    target = binary_cols[0]

print(f"\n🎯 Target Column Detected: {target}")

# ✅ Encode text columns → numbers
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col].astype(str))

print("\n🔄 All categorical columns converted to numeric")

# ✅ Split Data
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📦 Total Samples: {len(df)}")
print(f"🎓 Training Samples: {len(X_train)}")
print(f"🧪 Testing Samples: {len(X_test)}")

# ✅ Random Forest + Hyperparameter Tuning
params = {
    'n_estimators':[100,200],
    'max_depth':[5,10,20,None],
    'min_samples_split':[2,5],
    'min_samples_leaf':[1,2]
}

print("\n⚙️ Training & Hyperparameter Tuning...")
grid = GridSearchCV(RandomForestClassifier(), params, cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

model = grid.best_estimator_
print("✅ Best Parameters:", grid.best_params_)

# ✅ Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ✅ Evaluation
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("✅ Confusion Matrix:\n", cm)
print(f"🎯 Accuracy: {acc*100:.2f}%")

# ✅ Save the trained model as PKL file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n💾 Model saved successfully as model.pkl ✅")

# 📊 VISUALIZATIONS
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].hist(figsize=(15,10))
plt.suptitle("📊 Feature Histograms", fontsize=16)
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title("Feature Importance")
plt.show()

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.2f}")
plt.plot([0,1],[0,1],'--')
plt.title("ROC Curve")
plt.legend()
plt.show()

precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(recall, precision)
plt.title("Precision–Recall Curve")
plt.show()
