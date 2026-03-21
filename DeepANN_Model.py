# ============================================================
# DEEP ANN MODEL 
# ============================================================

!pip install rasterio --quiet

from google.colab import drive
drive.mount('/content/drive')

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

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
# 🔥 DEEP ANN MODEL (IMPROVED)
# ============================================================
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 🔥 EARLY STOPPING (IMPORTANT)
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

print("✅ DeepANN trained")

# ============================================================
# LOAD RASTERS
# ============================================================
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

out_path = "/content/drive/MyDrive/NUBRA/DeepANN_Map.tif"

# 🔥 LARGE BLOCK = FAST
block_size = 1024

# ============================================================
# PREDICTION LOOP (OPTIMIZED)
# ============================================================
with rasterio.open(out_path, "w", **meta) as dst:

    for row in range(0, rows, block_size):
        for col in range(0, cols, block_size):

            window = Window(col, row,
                            min(block_size, cols - col),
                            min(block_size, rows - row))

            stack = []

            for ds in datasets:
                arr = ds.read(1, window=window).astype(np.float32)

                if ds.nodata is not None:
                    arr[arr == ds.nodata] = np.nan

                arr = np.nan_to_num(arr, nan=0)
                stack.append(arr)

            stack = np.stack(stack, axis=-1)

            h, w, b = stack.shape
            pixels = stack.reshape(-1, b)

            # ---------------- SCALING ----------------
            pixels_scaled = scaler.transform(pixels)

            # 🔥 BATCH PREDICTION (VERY IMPORTANT)
            preds = model.predict(
                pixels_scaled,
                batch_size=8192,
                verbose=0
            ).flatten()

            pred_map = preds.reshape(h, w)

            dst.write(pred_map.astype(np.float32), 1, window=window)

print("✅ DeepANN susceptibility map generated")

# ============================================================
# VISUALIZATION
# ============================================================
with rasterio.open(out_path) as src:
    ann_map = src.read(1)

print("Prediction range:", ann_map.min(), ann_map.max())

plt.figure(figsize=(8,6))
plt.imshow(ann_map, cmap='RdYlBu_r', vmin=0, vmax=1)
plt.colorbar(label="Susceptibility")
plt.title("DeepANN Susceptibility Map")
plt.axis('off')
plt.show()
