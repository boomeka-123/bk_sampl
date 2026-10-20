import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

pd.set_option('display.width', 90)
pd.set_option('display.max_columns', None)

# ---------- 1. Create and load the transaction data ----------
rng = np.random.RandomState(42)
n = 20000

df = pd.DataFrame({
    'Amount': np.round(rng.lognormal(mean=3.8, sigma=1.0, size=n), 2),
    'Hour': rng.randint(0, 24, size=n),
    'Distance_From_Home': np.round(rng.exponential(scale=15, size=n), 2),
    'Txn_Last_24h': rng.poisson(2, size=n),
    'Is_Foreign': rng.binomial(1, 0.05, size=n),
    'Is_Online': rng.binomial(1, 0.35, size=n),
    'Account_Age_Days': rng.randint(30, 3000, size=n),
})

logit = (-13.0
         + 1.2 * np.log(df['Amount'])
         + 2.5 * (df['Hour'] < 6)
         + 0.05 * df['Distance_From_Home']
         + 0.5 * df['Txn_Last_24h']
         + 3.0 * df['Is_Foreign']
         + 1.0 * df['Is_Online']
         + 1.5 * (df['Account_Age_Days'] < 180))
prob = 1 / (1 + np.exp(-logit))
df['Is_Fraud'] = rng.binomial(1, prob)

print(df.head())
print("Shape:", df.shape)

# ---------- 2. Univariate analysis ----------
def univariate_analysis(df, column):
    freq = df[column].value_counts()
    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0]
    std = df[column].std()
    return freq, mean, median, mode, std

freq, mean, median, mode, std = univariate_analysis(df, 'Amount')
print("\nUnivariate Analysis for Amount")
print("Frequency (top 5):\n", freq.head())
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)
print("Standard Deviation:", std)

print("\nClass distribution (0 = Legit, 1 = Fraud):")
print(df['Is_Fraud'].value_counts())
print("Fraud percentage:", df['Is_Fraud'].mean() * 100)

plt.figure(figsize=(10, 4))
df['Is_Fraud'].value_counts().sort_index().plot(kind='bar', color=['tab:blue', 'tab:red'])
plt.title("Legit (0) vs Fraud (1) Transactions")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=0)
plt.show()

# ---------- 3. Features and target ----------
features = ['Amount', 'Hour', 'Distance_From_Home', 'Txn_Last_24h',
            'Is_Foreign', 'Is_Online', 'Account_Age_Days']
X = df[features]
y = df['Is_Fraud']

# ---------- 4. Stratified train-test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTotal, Train, Test:", len(df), len(X_train), len(X_test))
print("Fraud in Train, Test:", y_train.sum(), y_test.sum())

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ---------- 5. Logistic regression ----------
log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
log_reg.fit(X_train_sc, y_train)
pred_lr = log_reg.predict(X_test_sc)
prob_lr = log_reg.predict_proba(X_test_sc)[:, 1]

print("\nLogistic Regression")
print("Coefficients:", log_reg.coef_[0])
print("Intercept:", log_reg.intercept_[0])
print("F1 Score:", f1_score(y_test, pred_lr))

# ---------- 6. Decision tree ----------
tree = DecisionTreeClassifier(max_depth=6, class_weight='balanced', random_state=42)
tree.fit(X_train, y_train)
pred_dt = tree.predict(X_test)
prob_dt = tree.predict_proba(X_test)[:, 1]

print("\nDecision Tree")
print("F1 Score:", f1_score(y_test, pred_dt))

# ---------- 7. Random forest ----------
forest = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced',
                                random_state=42, n_jobs=1)
forest.fit(X_train, y_train)
pred_rf = forest.predict(X_test)
prob_rf = forest.predict_proba(X_test)[:, 1]

print("\nRandom Forest")
print("F1 Score:", f1_score(y_test, pred_rf))
print("Feature Importance:")
for name, imp in sorted(zip(features, forest.feature_importances_), key=lambda t: -t[1]):
    print("  ", name, ":", round(imp, 4))

# ---------- 8. Naive baseline ----------
naive_pred = np.zeros(len(y_test), dtype=int)
print("\nNaive baseline accuracy (predict everything as Legit):",
      accuracy_score(y_test, naive_pred))
print("Naive baseline recall:", recall_score(y_test, naive_pred))

# ---------- 9. Comparison ----------
results = {}
for name, pred, prob in [("Logistic Regression", pred_lr, prob_lr),
                         ("Decision Tree", pred_dt, prob_dt),
                         ("Random Forest", pred_rf, prob_rf)]:
    results[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, prob),
    }
results_df = pd.DataFrame(results).T
print("\nModel Comparison")
print(results_df.round(4))
best_name = results_df['F1'].idxmax()
print("Better model:", best_name)

# ---------- 10. Confusion matrix of the best model ----------
best_pred = {"Logistic Regression": pred_lr, "Decision Tree": pred_dt,
             "Random Forest": pred_rf}[best_name]
cm = confusion_matrix(y_test, best_pred)
print("\nConfusion Matrix (", best_name, ")")
print(cm)

plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix - " + best_name)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Legit", "Fraud"])
plt.yticks([0, 1], ["Legit", "Fraud"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.colorbar()
plt.show()

# ---------- 11. ROC curve ----------
plt.figure(figsize=(10, 4))
for name, prob in [("Logistic Regression", prob_lr), ("Decision Tree", prob_dt),
                   ("Random Forest", prob_rf)]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=name + " (AUC = " + str(round(roc_auc_score(y_test, prob), 3)) + ")")
plt.plot([0, 1], [0, 1], 'k--', label="Random guess")
plt.title("ROC Curve - Fraud Detection")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
