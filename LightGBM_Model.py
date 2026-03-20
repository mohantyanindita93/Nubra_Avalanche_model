
#  LIGHTGBM → SUSCEPTIBILITY MAP (OPTIMIZED)
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

import lightgbm as lgb
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

feature_names = list(X_train.columns)

# ---------------- TRAIN MODEL ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

lgb_model = lgb.LGBMClassifier(
    n_estimators=300,   # reduced for speed
    learning_rate=0.05,
    max_depth=8,
    n_jobs=-1           # 🔥 use all CPU cores
)

lgb_model.fit(X_train_scaled, y_train)

print("✅ LightGBM Model Trained")

# ---------------- LOAD RASTERS ----------------
all_rasters = [os.path.join(RASTER_PATH, f)
               for f in os.listdir(RASTER_PATH) if f.endswith(".tif")]

raster_dict = {os.path.basename(f).split('.')[0]: f for f in all_rasters}

ordered_files = [raster_dict[f] for f in feature_names]

datasets = [rasterio.open(f) for f in ordered_files]

ref = datasets[0]
rows, cols = ref.height, ref.width

# ---------------- OUTPUT ----------------
meta = ref.meta.copy()
meta.update(dtype=rasterio.float32, count=1)

out_path = "/content/drive/MyDrive/NUBRA/LGBM_FAST_Map.tif"

# 🔥 BIGGER BLOCK = FASTER
block_size = 512

with rasterio.open(out_path, "w", **meta) as dst:

    for row in range(0, rows, block_size):
        for col in range(0, cols, block_size):

            window = Window(col, row,
                            min(block_size, cols - col),
                            min(block_size, rows - row))

            stack = []

            for ds in datasets:
                arr = ds.read(1, window=window).astype(np.float32)

                nodata = ds.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan

                # Faster NaN handling
                arr = np.nan_to_num(arr, nan=0)

                stack.append(arr)

            stack = np.stack(stack, axis=-1)

            h, w, b = stack.shape
            pixels = stack.reshape(-1, b)

            # Scale
            pixels_scaled = scaler.transform(pixels)

            # Predict 
            pred = lgb_model.predict_proba(pixels_scaled)[:,1]

            pred_map = pred.reshape(h, w)

            dst.write(pred_map.astype(np.float32), 1, window=window)

print("✅ FAST LightGBM map generated")

# ---------------- VISUAL ----------------
with rasterio.open(out_path) as src:
    lgb_map = src.read(1)

plt.figure(figsize=(8,6))
plt.imshow(lgb_map, cmap='RdYlBu_r', vmin=0, vmax=1)
plt.colorbar(label="Susceptibility")
plt.title("LightGBM FAST Map")
plt.axis('off')
plt.show()
