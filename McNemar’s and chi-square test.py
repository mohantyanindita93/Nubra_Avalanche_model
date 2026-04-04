!pip install lightgbm statsmodels seaborn --quiet

# ---------------- IMPORTS ----------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2_contingency

# ---------------- LOAD DATA ----------------
train_df = pd.read_csv("/content/drive/MyDrive/NUBRA/Train_Nubra.csv")
test_df  = pd.read_csv("/content/drive/MyDrive/NUBRA/Test_Nubra.csv")

X_train = train_df.drop(columns=["Training_re"])
y_train = train_df["Training_re"]

X_test = test_df.drop(columns=["Testing_re"])
y_test = test_df["Testing_re"]

# ---------------- MODELS ----------------
models = {
    "LR": LogisticRegression(max_iter=1000, random_state=42),
    
    "MDA": LinearDiscriminantAnalysis(),
    
    "DeepANN": Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128,64,32),
            max_iter=1000,
            early_stopping=True,
            random_state=42))
    ]),
    
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        random_state=42
    ),
    
    "CNN+RF+CB": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
}

# ---------------- TRAIN + PREDICT ----------------
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

model_names = list(models.keys())
n = len(model_names)

# ---------------- MCNEMAR MATRIX ----------------
mcnemar_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            mcnemar_matrix[i, j] = 1
        else:
            pred1 = predictions[model_names[i]]
            pred2 = predictions[model_names[j]]

            both_correct = np.sum((pred1 == y_test) & (pred2 == y_test))
            both_wrong   = np.sum((pred1 != y_test) & (pred2 != y_test))
            b = np.sum((pred1 == y_test) & (pred2 != y_test))
            c = np.sum((pred1 != y_test) & (pred2 == y_test))

            table = [[both_correct, b],
                     [c, both_wrong]]

            result = mcnemar(table, exact=False, correction=True)
            mcnemar_matrix[i, j] = result.pvalue

# ---------------- CHI-SQUARE MATRIX ----------------
chi_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        if i == j:
            chi_matrix[i, j] = 1
        else:
            pred1 = predictions[model_names[i]]
            pred2 = predictions[model_names[j]]

            table = pd.crosstab(pred1, pred2)

            chi2, p, _, _ = chi2_contingency(table)
            chi_matrix[i, j] = p

# ---------------- PLOT SETTINGS ----------------
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

cmap = "RdYlBu_r"  # 🔥 best contrast for p-values

# ---------------- MCNEMAR HEATMAP ----------------
sns.heatmap(
    mcnemar_matrix,
    annot=True,
    fmt=".3f",
    cmap=cmap,
    vmin=0, vmax=1,
    xticklabels=model_names,
    yticklabels=model_names,
    annot_kws={"size": 13, "weight": "bold"},
    cbar_kws={"label": "p-value"},
    ax=axes[0]
)

axes[0].set_title("McNemar Test (p-values)", fontsize=18, weight="bold")

# ---------------- CHI-SQUARE HEATMAP ----------------
sns.heatmap(
    chi_matrix,
    annot=True,
    fmt=".3f",
    cmap=cmap,
    vmin=0, vmax=1,
    xticklabels=model_names,
    yticklabels=model_names,
    annot_kws={"size": 13, "weight": "bold"},
    cbar_kws={"label": "p-value"},
    ax=axes[1]
)

axes[1].set_title("Chi-square Test (p-values)", fontsize=18, weight="bold")

# ---------------- AXIS FORMATTING ----------------
for ax in axes:
    ax.tick_params(axis='x', labelsize=13, rotation=45)
    ax.tick_params(axis='y', labelsize=13)

plt.tight_layout()

# ---------------- SAVE HIGH QUALITY ----------------
plt.savefig(
    "/content/drive/MyDrive/NUBRA/Final_Statistical_Tests.png",
    dpi=1000,
    bbox_inches='tight'
)

plt.show()
