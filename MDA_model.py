# ============================================================
# MDA (LDA) → SUSCEPTIBILITY MAP (FINAL ONE CELL)
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

# ---------------- PATH ----------------
DATA_PATH = "/content/drive/MyDrive/NUBRA"
RASTER_PATH = "/content/drive/MyDrive/NUBRA/parameter_resampled"

train_file = os.path.join(DATA_PATH, "Train_Nubra.csv")

# ---------------- LOAD TRAIN ----------------
train_df = pd.read_csv(train_file)

target = "Training_re"

X_train = train_df.drop(columns=[target]).select_dtypes(include=[np.number])
y_train = train_df[target]

# 🔥 FEATURE ORDER
feature_names = list(X_train.columns)

print("Feature order used in model:")
print(feature_names)

# ---------------- TRAIN MODEL ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

mda = LinearDiscriminantAnalysis()
mda.fit(X_train_scaled, y_train)

print("✅ MDA Model Trained")

# ---------------- LOAD RASTERS ----------------
all_rasters = [os.path.join(RASTER_PATH, f) 
               for f in os.listdir(RASTER_PATH) if f.endswith(".tif")]

# 🔥 MATCH FEATURE ORDER
raster_dict = {os.path.basename(f).split('.')[0]: f for f in all_rasters}

ordered_files = []

for feat in feature_names:
    if feat in raster_dict:
        ordered_files.append(raster_dict[feat])
    else:
        raise ValueError(f"❌ Missing raster for feature: {feat}")

print("\nRaster order matched successfully")

datasets = [rasterio.open(f) for f in ordered_files]

ref = datasets[0]
rows, cols = ref.height, ref.width

# ---------------- OUTPUT ----------------
meta = ref.meta.copy()
meta.update(dtype=rasterio.float32, count=1)

out_path = "/content/drive/MyDrive/NUBRA/MDA_Susceptibility_Map.tif"

# ---------------- WINDOW PROCESSING ----------------
block_size = 128

with rasterio.open(out_path, "w", **meta) as dst:

    for row in range(0, rows, block_size):
        for col in range(0, cols, block_size):

            window = Window(col, row,
                            min(block_size, cols - col),
                            min(block_size, rows - row))

            stack = []

            for ds in datasets:
                arr = ds.read(1, window=window).astype(np.float64)

                # Handle NoData
                nodata = ds.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan

                # Replace NaN
                arr = np.nan_to_num(arr, nan=np.nanmean(arr))

                # Clip extreme values
                arr = np.clip(arr, -1e6, 1e6)

                stack.append(arr)

            stack = np.stack(stack, axis=-1)

            h, w, b = stack.shape
            pixels = stack.reshape(-1, b)

            # Scale
            pixels_scaled = scaler.transform(pixels)

            pixels_scaled = np.nan_to_num(pixels_scaled, nan=0, posinf=0, neginf=0)

            # ---------------- PREDICT ----------------
            pred = mda.predict_proba(pixels_scaled)[:,1]

            pred_map = pred.reshape(h, w)

            dst.write(pred_map.astype(np.float32), 1, window=window)

print("✅ MDA susceptibility map generated")

# ---------------- VISUAL CHECK ----------------
with rasterio.open(out_path) as src:
    mda_map = src.read(1)

print("Prediction range:", mda_map.min(), mda_map.max())

plt.figure(figsize=(8,6))
plt.imshow(mda_map, cmap='RdYlBu_r', vmin=0, vmax=1)
plt.colorbar(label="Susceptibility")
plt.title("MDA Avalanche Susceptibility Map")
plt.axis('off')
plt.show()
