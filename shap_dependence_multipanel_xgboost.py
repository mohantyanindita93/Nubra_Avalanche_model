# ============================================================
# CLEAN SHAP MULTI-PANEL (NO OVERLAP, FINAL PUBLICATION)
# ============================================================

!pip install xgboost shap --quiet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
import string

# ---------------- LOAD DATA ----------------
df = pd.read_csv("/content/drive/MyDrive/NUBRA/Train_Nubra.csv")

target = "Training_re"

exclude_cols = [
    target,
    "geology", "Landuse_La",
    "Profile_Cu", "Plan_Curva"
]

X = df.drop(columns=[col for col in exclude_cols if col in df.columns])
X = X.select_dtypes(include=[np.number])
y = df[target]

feature_names = X.columns.tolist()

# ---------------- TRAIN MODEL ----------------
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    n_jobs=-1
)

model.fit(X, y)

# ---------------- SHAP ----------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ---------------- PLOT SETTINGS ----------------
cols = 3
rows = int(np.ceil(len(feature_names) / cols))

fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 4))
axes = axes.flatten()

letters = list(string.ascii_uppercase)

# ---------------- PLOTTING ----------------
for i, feature in enumerate(feature_names):

    ax = axes[i]

    x_vals = X[feature]
    y_vals = shap_values[:, i]

    sc = ax.scatter(
        x_vals, y_vals,
        c=y_vals,
        cmap='coolwarm',
        s=20,
        edgecolor='black',
        linewidth=0.3,
        alpha=1.0
    )

    # 🔥 CLEAN TITLES (SHORT)
    ax.set_title(feature, fontsize=10, pad=6)

    # 🔥 SMALL LABELS
    ax.set_xlabel("Value", fontsize=8)
    ax.set_ylabel("SHAP", fontsize=8)

    # 🔥 PANEL LABEL
    ax.text(
        0.92, 0.88,
        letters[i],
        transform=ax.transAxes,
        fontsize=11,
        fontweight='bold'
    )

    # 🔥 TICK SIZE REDUCED
    ax.tick_params(axis='both', labelsize=7)

# Remove extra axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# ---------------- SPACING FIX (MOST IMPORTANT) ----------------
plt.subplots_adjust(
    left=0.06,
    right=0.85,   # space for colorbar
    top=0.95,
    bottom=0.06,
    wspace=0.30,  # horizontal spacing
    hspace=0.35   # vertical spacing
)

# ---------------- COLORBAR (NO OVERLAP) ----------------
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
cbar = fig.colorbar(sc, cax=cbar_ax)
cbar.set_label("SHAP value", fontsize=10)

# ---------------- SAVE ----------------
plt.savefig(
    "/content/drive/MyDrive/NUBRA/SHAP_CLEAN_FINAL.png",
    dpi=800,
    bbox_inches='tight'
)

plt.show()
