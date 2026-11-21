
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger
)

from importlib.util import spec_from_file_location


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# TRAINING CONFIGURATION
# ==========================================

EPOCHS = 30
LEARNING_RATE = 0.0001
NUM_CLASSES = 4

BEST_MODEL_PATH = MODEL_DIR / "custom_cnn_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "custom_cnn_final.keras"
LOG_FILE = LOG_DIR / "custom_cnn_training.csv"


# ==========================================
# LOAD DATA LOADER
# ==========================================

loader_path = SRC_DIR / "02_data_loader.py"

spec = spec_from_file_location(
    "data_loader",
    loader_path
)

data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)

train_dataset = data_loader.train_dataset
val_dataset = data_loader.val_dataset
y_train = data_loader.y_train


# ==========================================
# CALCULATE CLASS WEIGHTS
# ==========================================

class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

total_samples = len(y_train)

class_weights = {}

for class_id in range(NUM_CLASSES):

    class_weights[class_id] = (
        total_samples /
        (NUM_CLASSES * class_counts[class_id])
    )

print("\nClass Counts:")
print(class_counts)

print("\nClass Weights:")
print(class_weights)


# ==========================================
# LOAD CUSTOM CNN ARCHITECTURE
# ==========================================

from importlib.util import spec_from_file_location

architecture_path = SRC_DIR / "04_cnn_architecture.py"

architecture_spec = spec_from_file_location(
    "cnn_architecture",
    architecture_path
)

architecture_module = importlib.util.module_from_spec(
    architecture_spec
)

architecture_spec.loader.exec_module(
    architecture_module
)

model = architecture_module.build_custom_cnn()


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


# ==========================================
# TRAINING CALLBACKS
# ==========================================

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.0000001,
    verbose=1
)

csv_logger = CSVLogger(
    filename=LOG_FILE
)


# ==========================================
# START MODEL TRAINING
# ==========================================

print("\nStarting Custom CNN Training...\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
        csv_logger
    ]
)


# ==========================================
# SAVE FINAL MODEL
# ==========================================

model.save(FINAL_MODEL_PATH)

print("\nTraining Completed!")

print("Best Model:", BEST_MODEL_PATH)
print("Final Model:", FINAL_MODEL_PATH)
print("Training Log:", LOG_FILE)
