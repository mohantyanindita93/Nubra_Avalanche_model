# ============================================================
# FINAL: LR+MDA (1 plot) + others separate (ONE CELL)
# ============================================================

!pip install lightgbm catboost --quiet

from google.colab import drive
drive.mount('/content/drive')

import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ============================================================
# FIX RANDOMNESS
# ============================================================
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# ============================================================
# LOAD DATA
# ============================================================
path = "/content/drive/MyDrive/NUBRA/"

train_df = pd.read_csv(path + "Train_Nubra.csv")
test_df  = pd.read_csv(path + "Test_Nubra.csv")

train_target = "Training_re"
test_target  = "Testing_re"

X_train = train_df.drop(columns=[train_target]).select_dtypes(include=[np.number])
y_train = train_df[train_target]

X_test = test_df.drop(columns=[test_target]).select_dtypes(include=[np.number])
y_test = test_df[test_target]

# ============================================================
# SCALING (ONLY FOR ANN)
# ============================================================
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ============================================================
# FUNCTION FOR ROC
# ============================================================
def plot_single(y_true, prob, title):
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

# ============================================================
# 1. LR + MDA (SAME PLOT)
# ============================================================
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
prob_lr = lr.predict_proba(X_test)[:,1]

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
prob_lda = lda.predict_proba(X_test)[:,1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
fpr_lda, tpr_lda, _ = roc_curve(y_test, prob_lda)

plt.figure(figsize=(6,5))
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc(fpr_lr,tpr_lr):.4f})")
plt.plot(fpr_lda, tpr_lda, label=f"MDA (AUC={auc(fpr_lda,tpr_lda):.4f})")
plt.plot([0,1],[0,1],'k--')

plt.title("ROC Curve Comparison - LR vs MDA")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# ============================================================
# 2. LightGBM (SEPARATE)
# ============================================================
lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train)
prob_lgb = lgb_model.predict_proba(X_test)[:,1]

plot_single(y_test, prob_lgb, "ROC Curve - LightGBM")

# ============================================================
# 3. DeepANN (SEPARATE)
# ============================================================
ann = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

ann.compile(optimizer='adam', loss='binary_crossentropy')
ann.fit(X_train_s, y_train, epochs=30, batch_size=64, verbose=0)

prob_ann = ann.predict(X_test_s, verbose=0).flatten()

plot_single(y_test, prob_ann, "ROC Curve - Deep ANN")

# ============================================================
# 4. CNN + RF + CatBoost (HYBRID)
# ============================================================
# CNN
cnn = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer='adam', loss='binary_crossentropy')
cnn.fit(X_train_s, y_train, epochs=15, verbose=0)

prob_cnn = cnn.predict(X_test_s, verbose=0).flatten()

# RF
rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
rf.fit(X_train, y_train)
prob_rf = rf.predict_proba(X_test)[:,1]

# CatBoost
cb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05,
                        verbose=0, random_seed=42)
cb.fit(X_train, y_train)
prob_cb = cb.predict_proba(X_test)[:,1]

# Hybrid
prob_hybrid = (prob_cnn + prob_rf + prob_cb) / 3

plot_single(y_test, prob_hybrid, "ROC Curve - CNN + RF + CatBoost")

# ============================================================
# ALL MODEL METRICS TABLE (LIKE YOUR PAPER)
# ============================================================

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, matthews_corrcoef,
    roc_auc_score
)

def compute_metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, pred)
    accuracy = accuracy_score(y_true, pred)
    f1 = f1_score(y_true, pred)
    mcc = matthews_corrcoef(y_true, pred)
    roc_auc = roc_auc_score(y_true, prob)
    
    return [
        sensitivity, specificity, precision,
        accuracy, f1, mcc, roc_auc
    ]

# ============================================================
# COMPUTE FOR ALL MODELS
# (USE YOUR PROBABILITIES FROM PREVIOUS CODE)
# ============================================================

results = pd.DataFrame(columns=[
    "Model", "Sensitivity", "Specificity",
    "Precision", "Accuracy", "F1 Score",
    "MCC", "ROC-AUC"
])

results.loc[len(results)] = ["LR", *compute_metrics(y_test, prob_lr)]
results.loc[len(results)] = ["MDA (LDA)", *compute_metrics(y_test, prob_lda)]
results.loc[len(results)] = ["DeepANN", *compute_metrics(y_test, prob_ann)]
results.loc[len(results)] = ["LightGBM", *compute_metrics(y_test, prob_lgb)]
results.loc[len(results)] = ["CNN + RF + CatBoost", *compute_metrics(y_test, prob_hybrid)]

# ============================================================
# ROUND + DISPLAY
# ============================================================

results = results.round(4)

print("\n📊 FINAL MODEL PERFORMANCE TABLE:\n")
print(results)
