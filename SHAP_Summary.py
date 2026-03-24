# ============================================================
# SHAP ANALYSIS (XGBOOST) - NUBRA DATASET 
# ============================================================

# ---------------- MOUNT DRIVE ----------------
from google.colab import drive
drive.mount('/content/drive')

# ---------------- INSTALL LIBRARIES ----------------
!pip install shap xgboost --quiet

# ---------------- IMPORT LIBRARIES ----------------
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# ---------------- LOAD DATA ----------------
file_path = "/content/drive/MyDrive/NUBRA/Train_Nubra.csv"

df = pd.read_csv(file_path)

# ---------------- DEFINE TARGET ----------------
target = "Training_re"

# ---------------- PREPARE DATA ----------------
X = df.drop(columns=[target])
y = df[target]

print("Dataset Shape:", X.shape)

# ---------------- TRAIN XGBOOST MODEL ----------------
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X, y)

print("Model trained successfully!")

# ---------------- SHAP EXPLAINER ----------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For binary classification (safety check)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# ---------------- SHAP SUMMARY (DOT PLOT) ----------------
plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X,
    plot_type="dot",
    show=False
)

plt.title("SHAP Summary Plot (XGBoost)", fontsize=14)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/NUBRA/SHAP_XGB_summary_dot.png", dpi=600)
plt.show()

# ---------------- SHAP BAR PLOT ----------------
plt.figure(figsize=(8,6))

shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance (XGBoost)", fontsize=14)
plt.tight_layout()
plt.savefig("/content/drive/MyDrive/NUBRA/SHAP_XGB_summary_bar.png", dpi=600)
plt.show()

# ---------------- SHAP DEPENDENCE PLOTS ----------------
# Select important features manually if needed
features_to_plot = X.columns[:5]

for feature in features_to_plot:
    plt.figure(figsize=(6,4))
    shap.dependence_plot(feature, shap_values, X, show=False)
    plt.title(f"SHAP Dependence: {feature}")
    plt.tight_layout()
    plt.savefig(f"/content/drive/MyDrive/NUBRA/SHAP_XGB_dependence_{feature}.png", dpi=600)
    plt.show()

# ---------------- SAVE SHAP VALUES ----------------
shap_df = pd.DataFrame(shap_values, columns=X.columns)
shap_df.to_csv("/content/drive/MyDrive/NUBRA/SHAP_XGB_values.csv", index=False)

# ---------------- FEATURE IMPORTANCE TABLE ----------------
importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:\n", importance_df.head(10))

importance_df.to_csv("/content/drive/MyDrive/NUBRA/SHAP_XGB_feature_importance.csv", index=False)

print("\n✅ SHAP (XGBoost) analysis completed and saved to Drive!")
