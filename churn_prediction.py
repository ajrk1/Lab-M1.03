"""
Telco Customer Churn Prediction with KNN
Author: [Anson Knausenberger]
Description: Predict whether a telecom customer will churn (Yes/No) using KNN.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
print("Loading Telco churn dataset...")
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("\nShape (rows, columns):", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values (top 20):")
missing = df.isnull().sum().sort_values(ascending=False)
print(missing.head(20))

print("\nTarget distribution (Churn):")
print(df["Churn"].value_counts())
print("\nChurn rate (%):")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))
print("\nCreating quick visualizations...")

# 1) Churn count plot
churn_counts = df["Churn"].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(churn_counts.index, churn_counts.values)
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("churn_distribution.png", dpi=150)
plt.show()

# 2) MonthlyCharges distribution by churn
plt.figure(figsize=(8, 5))
plt.hist(df.loc[df["Churn"] == "No", "MonthlyCharges"], bins=30, alpha=0.5, label="No")
plt.hist(df.loc[df["Churn"] == "Yes", "MonthlyCharges"], bins=30, alpha=0.5, label="Yes")
plt.title("MonthlyCharges Distribution by Churn")
plt.xlabel("MonthlyCharges")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("monthlycharges_by_churn.png", dpi=150)
plt.show()

print("Saved: churn_distribution.png and monthlycharges_by_churn.png")
print("\n" + "="*50)
print("STEP 2: PREPROCESSING")
print("="*50)

df_clean = df.copy()

# 1) Drop customerID (identifier, not useful for prediction)
df_clean = df_clean.drop(columns=["customerID"])

# 2) Convert TotalCharges to numeric (it is currently a string)
# Some rows may have blanks/spaces -> turn into NaN, then fill
df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

print("Missing values after TotalCharges conversion:")
print(df_clean.isnull().sum().sort_values(ascending=False).head(10))

# Fill missing TotalCharges with the median (simple, reasonable default)
df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median())

# 3) Convert target to binary: Yes=1, No=0
df_clean["Churn"] = df_clean["Churn"].map({"No": 0, "Yes": 1})

# 4) One-hot encode categorical columns
df_encoded = pd.get_dummies(df_clean, drop_first=True)

print("\nEncoded shape (rows, columns):", df_encoded.shape)

# 5) Split X/y
X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nTarget distribution after encoding:")
print(y.value_counts())
print("Churn rate (%):", round(y.mean() * 100, 2))
print("\n" + "="*50)
print("STEP 3: TRAIN/TEST SPLIT")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

print("\nTrain churn distribution:")
print(y_train.value_counts())
print("Train churn rate (%):", round(y_train.mean() * 100, 2))

print("\nTest churn distribution:")
print(y_test.value_counts())
print("Test churn rate (%):", round(y_test.mean() * 100, 2))
print("\n" + "="*50)
print("STEP 4: TRAIN KNN (k=5)")
print("="*50)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

print("Trained KNN with k=5")
print("Train predictions:", len(y_train_pred))
print("Test predictions:", len(y_test_pred))
print("\n" + "="*50)
print("STEP 5: EVALUATION (positive = churn=1)")
print("="*50)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

test_prec = precision_score(y_test, y_test_pred)  # positive class = 1 (churn)
test_rec = recall_score(y_test, y_test_pred)      # positive class = 1 (churn)

cm = confusion_matrix(y_test, y_test_pred)

print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
print(f"Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test Precision (churn): {test_prec:.4f}")
print(f"Test Recall (churn):    {test_rec:.4f}")

print("\nConfusion Matrix (rows=Actual, cols=Predicted)")
print("Order: 0=No churn, 1=Churn")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["No", "Yes"]))
print("\n" + "="*50)
print("STEP 6: K COMPARISON")
print("="*50)

k_values = [1, 3, 5, 7, 9, 11, 15]
results = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)   # churn=1
    rec = recall_score(y_test, preds)       # churn=1

    results.append({"K": k, "Accuracy": acc, "Precision": prec, "Recall": rec})

    print(f"K={k:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

results_df = pd.DataFrame(results)
print("\nResults table:")
print(results_df)

best_by_recall = results_df.loc[results_df["Recall"].idxmax(), "K"]
print(f"\nBest K by Recall (catching churners): {best_by_recall}")

best_by_accuracy = results_df.loc[results_df["Accuracy"].idxmax(), "K"]
print(f"Best K by Accuracy: {best_by_accuracy}")

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("\n" + "="*50)
print("OPTIONAL UPGRADE: PIPELINE (SCALE NUMERIC + OHE CATEGORICAL)")
print("="*50)

# Start again from the original df (before we manually encoded)
df_model = df.copy()

# Drop ID
df_model = df_model.drop(columns=["customerID"])

# Convert TotalCharges to numeric + fill missing
df_model["TotalCharges"] = pd.to_numeric(df_model["TotalCharges"], errors="coerce")
df_model["TotalCharges"] = df_model["TotalCharges"].fillna(df_model["TotalCharges"].median())

# Target
y2 = df_model["Churn"].map({"No": 0, "Yes": 1})
X2 = df_model.drop(columns=["Churn"])

# Identify numeric vs categorical columns
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical_features = [c for c in X2.columns if c not in numeric_features]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

k_values = [1, 3, 5, 7, 9, 11, 15]
pipe_results = []

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

for k in k_values:
    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    model.fit(X2_train, y2_train)
    preds = model.predict(X2_test)

    acc = accuracy_score(y2_test, preds)
    prec = precision_score(y2_test, preds)
    rec = recall_score(y2_test, preds)

    pipe_results.append({"K": k, "Accuracy": acc, "Precision": prec, "Recall": rec})
    print(f"[Pipeline] K={k:2d}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")

pipe_df = pd.DataFrame(pipe_results)
print("\nPipeline results table:")
print(pipe_df)

print("\nBest K by Recall (pipeline):", pipe_df.loc[pipe_df["Recall"].idxmax(), "K"])
print("Best K by Accuracy (pipeline):", pipe_df.loc[pipe_df["Accuracy"].idxmax(), "K"])


# ==================================================
# STEP 7: ANALYSIS & RECOMMENDATIONS
# ==================================================
# Model performance (baseline KNN):
# - Best overall accuracy in our K sweep was ~0.789 (K=9 or K=15), but recall for churners dropped (~0.43 to 0.40).
# - Best recall for churners was ~0.455 (K=1), but precision and accuracy were notably lower.
# - A reasonable balance was K=7 (Accuracy ~0.781, Precision ~0.626, Recall ~0.439).
#
# Business interpretation:
# - The model misses more than half of actual churners (recall < 0.46 across all K tested),
#   meaning many at-risk customers would NOT be flagged for retention outreach.
# - False alarms exist (precision ~0.46–0.67 depending on K), so outreach would also include some non-churners.
#
# What features seem important (quick EDA insight):
# - MonthlyCharges appears higher for churners than non-churners based on the histogram.
# - (Optional next checks: compare churn by Contract type, tenure, InternetService, PaymentMethod.)
#
# Recommendations:
# 1) Use this KNN model only as a simple baseline.
# 2) Improve performance by scaling features (KNN relies on distance) and trying stronger models
#    like Logistic Regression, Random Forest, or Gradient Boosting.
# 3) Align the decision threshold and metric with business goals (often prioritize recall for churn).
#
# Limitations:
# - KNN is sensitive to feature scaling and high-dimensional one-hot encoded data.
# - We did not tune features deeply or scale numeric columns, which can reduce KNN performance.
# - Accuracy is not the best metric with class imbalance; focus on recall/precision/F1 for churners.
#
# Upgrade result (recommended):
# - Using a proper preprocessing pipeline (scaling numeric features + one-hot encoding categoricals) improved churn recall substantially.
# - Best pipeline result was around K=9 with Accuracy ~0.78 and Recall ~0.586 for churners (vs ~0.43–0.46 without scaling).
# - Recommendation: use the pipeline version as your final KNN baseline.
