# ============================================================
# HYBRID MODEL (CNN + RF + CATBOOST) for avalanche susceptibility mapping
# ============================================================

# 🔥 INSTALL MISSING LIBRARIES
!pip install catboost --quiet

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ============================================================
# 1. CNN (LIGHT VERSION)
# ============================================================
cnn_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

cnn_model.fit(X_train_scaled, y_train,
              epochs=15,   # 🔥 reduced
              batch_size=32,
              verbose=0)

print("✅ CNN trained")

# ============================================================
# 2. RANDOM FOREST (LIGHT)
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)

print("✅ RF trained")

# ============================================================
# 3. CATBOOST (LIGHT VERSION)
# ============================================================
cb_model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    verbose=0
)

cb_model.fit(X_train, y_train)

print("✅ CatBoost trained")

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

out_path = "/content/drive/MyDrive/NUBRA/Hybrid_FAST_Map.tif"

# 🔥 BIG BLOCK → FAST + LOW RAM
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

                arr = np.nan_to_num(arr, nan=0)

                stack.append(arr)

            stack = np.stack(stack, axis=-1)

            h, w, b = stack.shape
            pixels = stack.reshape(-1, b)

            # ---------------- MODEL PREDICTIONS ----------------

            # CNN (scaled)
            pixels_scaled = scaler.transform(pixels)
            pixels_scaled = np.nan_to_num(pixels_scaled)

            cnn_pred = cnn_model.predict(pixels_scaled, verbose=0).flatten()

            # RF
            rf_pred = rf_model.predict_proba(pixels)[:,1]

            # CatBoost
            cb_pred = cb_model.predict_proba(pixels)[:,1]

            # ---------------- HYBRID ----------------
            hybrid_pred = (cnn_pred + rf_pred + cb_pred) / 3.0

            pred_map = hybrid_pred.reshape(h, w)

            dst.write(pred_map.astype(np.float32), 1, window=window)

print("✅ Hybrid map generated (FAST + STABLE)")

# ---------------- VISUAL ----------------
with rasterio.open(out_path) as src:
    hybrid_map = src.read(1)

print("Prediction range:", hybrid_map.min(), hybrid_map.max())

plt.figure(figsize=(8,6))
plt.imshow(hybrid_map, cmap='RdYlBu_r', vmin=0, vmax=1)
plt.colorbar(label="Susceptibility")
plt.title("Hybrid (CNN+RF+CB) FAST Map")
plt.axis('off')
plt.show()
