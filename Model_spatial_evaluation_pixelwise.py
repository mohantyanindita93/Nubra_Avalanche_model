# ============================================================
# CORRECT SPATIAL ROC (PIXEL-WISE)
# ============================================================

!pip install rasterio geopandas --quiet

from google.colab import drive
drive.mount('/content/drive')

import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from rasterio.features import geometry_mask
from sklearn.metrics import roc_curve, auc

# ---------------- PATH ----------------
base = "/content/drive/MyDrive/NUBRA/model_output/"

rasters = {
    "LR": base + "LR.tif",
    "MDA": base + "MDA.tif",
    "LightGBM": base + "lgb.tif",
    "DeepANN": base + "deepann1.tif",
    "Hybrid": base + "cnn+rf+cb.tif"
}

shp = base + "avl_poly.shp"

# ---------------- LOAD SHAPE ----------------
gdf = gpd.read_file(shp)

plt.figure(figsize=(8,6))

# ============================================================
# LOOP FOR EACH MODEL
# ============================================================
for name, path in rasters.items():

    with rasterio.open(path) as src:

        # Reproject polygon
        if gdf.crs != src.crs:
            gdf_proj = gdf.to_crs(src.crs)
        else:
            gdf_proj = gdf

        # Read raster
        data = src.read(1)
        data = data.astype(float)

        # Remove nodata
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        # Create mask (True = outside)
        mask = geometry_mask(
            gdf_proj.geometry,
            transform=src.transform,
            invert=True,
            out_shape=data.shape
        )

        # Flatten
        data_flat = data.flatten()
        mask_flat = mask.flatten()

        # Remove NaN
        valid = ~np.isnan(data_flat)
        data_flat = data_flat[valid]
        mask_flat = mask_flat[valid]

        # Labels
        y_true = mask_flat.astype(int)   # 1 = avalanche, 0 = non

        # Scores
        y_score = data_flat

        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

# Random line
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curves for Avalanche Susceptibility Models")
plt.legend()
plt.grid()
plt.show()
