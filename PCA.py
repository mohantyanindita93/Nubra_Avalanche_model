# ============================================================
# PCA MULTI-PANEL (DEM VARIABLES ONLY) WITH VALUES + LEGENDS
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

# ---------------- IMPORT ----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_style("white")
sns.set_context("paper", font_scale=1.2)

# ---------------- LOAD DATA ----------------
file_path = "/content/drive/MyDrive/NUBRA/Train_Nubra.csv"
df = pd.read_csv(file_path)

# ---------------- DEM VARIABLES ----------------
dem_vars = [
    "Slope_deg", "Aspect", "Curvature",
    "Plan_Curva", "Profile_Cu",
    "TPI", "TRI", "TWI",
    "Elevation", "Relief_Amp"
]

X = df[dem_vars]

# ---------------- STANDARDIZE ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- PCA ----------------
pca = PCA(n_components=len(dem_vars))
X_pca = pca.fit_transform(X_scaled)

explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

# ---------------- LOADINGS ----------------
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(dem_vars))],
    index=dem_vars
).round(2)

# ---------------- CORRELATION ----------------
corr_before = pd.DataFrame(X_scaled, columns=dem_vars).corr().round(2)
corr_after = pd.DataFrame(X_pca).corr().round(2)

# ---------------- VIF ----------------
vif_vals = [variance_inflation_factor(X_pca, i) for i in range(X_pca.shape[1])]
vif_df = pd.DataFrame({
    "PC": [f'PC{i+1}' for i in range(len(dem_vars))],
    "VIF": np.round(vif_vals, 2)
})

# ============================================================
# ---------------- PLOT ----------------
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# -------- (A) CORRELATION BEFORE PCA --------
sns.heatmap(
    corr_before,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Correlation Coefficient"},
    ax=axes[0, 0],
    annot_kws={"size":7}
)
axes[0, 0].set_title("(A) Correlation BEFORE PCA")

# -------- (B) SCREE PLOT --------
bars = axes[0, 1].bar(range(1, len(dem_vars)+1), explained_var, label="Individual Variance")
line = axes[0, 1].plot(range(1, len(dem_vars)+1), cum_var, color='red', marker='o', label="Cumulative Variance")

# Value labels
for i, v in enumerate(explained_var):
    axes[0, 1].text(i+1, v+0.01, f"{v:.2f}", ha='center', fontsize=7)

for i, v in enumerate(cum_var):
    axes[0, 1].text(i+1, v+0.03, f"{v:.2f}", ha='center', fontsize=7)

axes[0, 1].axhline(0.90, linestyle='--', color='green', label="90% Threshold")
axes[0, 1].set_title("(B) Scree Plot")
axes[0, 1].set_xlabel("Principal Components")
axes[0, 1].set_ylabel("Explained Variance")
axes[0, 1].legend()

# -------- (C) PCA LOADINGS --------
sns.heatmap(
    loadings,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Loading Value"},
    ax=axes[0, 2],
    annot_kws={"size":7}
)
axes[0, 2].set_title("(C) PCA Loadings")

# -------- (D) CORRELATION AFTER PCA --------
sns.heatmap(
    corr_after,
    annot=True,
    fmt=".1e",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Correlation Coefficient"},
    ax=axes[1, 0],
    annot_kws={"size":6}
)
axes[1, 0].set_title("(D) Correlation AFTER PCA")

# -------- (E) VIF --------
bars = axes[1, 1].barh(vif_df["PC"], vif_df["VIF"], color='steelblue')

# Value labels
for i, v in enumerate(vif_df["VIF"]):
    axes[1, 1].text(v + 0.02, i, f"{v:.2f}", va='center', fontsize=8)

axes[1, 1].set_title("(E) VIF AFTER PCA (~1)")
axes[1, 1].set_xlabel("VIF")

# -------- EMPTY PANEL --------
axes[1, 2].axis("off")

# ---------------- FINAL LAYOUT ----------------
plt.tight_layout()

# ---------------- SAVE ----------------
plt.savefig(
    "/content/drive/MyDrive/NUBRA/PCA_DEM_Final.png",
    dpi=1000,
    bbox_inches='tight'
)

plt.show()

print("✅ FINAL PCA DEM figure saved (1000 DPI, with values + legends)")

# ===============================
# FINAL PCA LOLLIPOP (NO TEXT OVERLAP)
# ===============================
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

fig, axes = plt.subplots(5, 2, figsize=(18, 22))
axes = axes.flatten()

cmap = plt.cm.coolwarm
norm = plt.Normalize(-1, 1)

for i, pc in enumerate(loadings.columns):
    ax = axes[i]
    values = loadings[pc].sort_values()

    y_pos = np.arange(len(values))

    # stems
    ax.hlines(y=y_pos, xmin=0, xmax=values,
              color='gray', linewidth=1.5)

    # points
    colors = cmap(norm(values))
    ax.scatter(values, y_pos, c=colors, s=90,
               edgecolor='black', linewidth=0.7, zorder=3)

    # y labels (bigger font)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(values.index, fontsize=11)

    # title
    ax.set_title(pc, fontsize=14, weight='bold')

    # -------------------------------
    # VALUE LABELS (NO OVERLAP FIX)
    # -------------------------------
    for j, val in enumerate(values):
        # dynamic offset (auto spacing)
        if val >= 0:
            offset = 0.05
            ha = 'left'
        else:
            offset = -0.08
            ha = 'right'

        ax.text(val + offset, j,
                f"{val:.3f}",
                va='center',
                ha=ha,
                fontsize=10,
                fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))

    # center line
    ax.axvline(0, color='black', linewidth=1)

    ax.set_xlim(-1, 1)
    ax.tick_params(axis='x', labelsize=10)

# ===============================
# LAYOUT FIX
# ===============================
plt.subplots_adjust(
    left=0.12,
    right=0.98,
    top=0.96,
    bottom=0.10,
    hspace=0.5,
    wspace=0.3
)

# ===============================
# COLORBAR (NO OVERLAP)
# ===============================
cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.02])

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
cbar.set_label("Loading Value (Negative → Positive)", fontsize=13)

# ===============================
# SAVE (HIGH QUALITY)
# ===============================
plt.savefig(
    "/content/drive/MyDrive/NUBRA/PCA_Lollipop_Final_Clean.png",
    dpi=1000,
    bbox_inches='tight'
)

plt.show()
