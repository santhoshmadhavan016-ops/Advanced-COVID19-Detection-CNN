
import importlib.util
import traceback
from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight


# ==================================================
# 1. CPU CONFIGURATION
# ==================================================

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

print("\nTraining device: CPU", flush=True)


# ==================================================
# 2. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# New filenames: preserve the previous checkpoint
BEST_WEIGHTS_PATH = MODEL_DIR / "densenet121_best_v2.weights.h5"
FINAL_WEIGHTS_PATH = MODEL_DIR / "densenet121_final_v2.weights.h5"

CSV_LOG_PATH = LOG_DIR / "densenet121_training_v2.csv"


# ==================================================
# 3. LOAD DATASET
# ==================================================

def load_dataset():

    print("\n[STEP 1] Loading dataset...", flush=True)

    loader_path = PROJECT_ROOT / "src" / "02_data_loader.py"

    if not loader_path.exists():
        raise FileNotFoundError(
            f"Data loader not found: {loader_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "data_loader",
        loader_path
    )

    data_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader)

    train_dataset = data_loader.train_dataset
    val_dataset = data_loader.val_dataset

    y_train = np.asarray(data_loader.y_train)

    print("Training samples:", len(y_train), flush=True)
    print(
        "Validation samples:",
        len(data_loader.y_val),
        flush=True
    )

    print("Training dataset:", train_dataset.element_spec, flush=True)
    print("Validation dataset:", val_dataset.element_spec, flush=True)

    return train_dataset, val_dataset, y_train


# ==================================================
# 4. CALCULATE CLASS WEIGHTS
# ==================================================

def calculate_class_weights(y_train):

    print("\n[STEP 2] Calculating class weights...", flush=True)

    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(classes, weights)
    }

    print("\nClass weights:", flush=True)
    print(class_weights, flush=True)

    return class_weights


# ==================================================
# 5. LOAD DENSENET121 ARCHITECTURE
# ==================================================

def load_model_architecture():

    print("\n[STEP 3] Building DenseNet121...", flush=True)

    architecture_path = (
        PROJECT_ROOT / "src" / "11_densenet121_architecture.py"
    )

    if not architecture_path.exists():
        raise FileNotFoundError(
            f"DenseNet121 architecture not found: {architecture_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "densenet_architecture",
        architecture_path
    )

    architecture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(architecture)

    model = architecture.build_densenet121()

    print("\nDenseNet121 model built successfully!", flush=True)
    print("Input shape:", model.input_shape, flush=True)
    print("Output shape:", model.output_shape, flush=True)
    print("Total parameters:", model.count_params(), flush=True)
    print(
        "Trainable parameters:",
        sum(
            np.prod(variable.shape)
            for variable in model.trainable_variables
        ),
        flush=True
    )

    return model


# ==================================================
# 6. TRAINING CALLBACKS
# ==================================================

def create_callbacks():

    print("\n[STEP 4] Preparing callbacks...", flush=True)

    callbacks = [

        # Save only the best model weights
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_WEIGHTS_PATH),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            verbose=1
        ),

        # Stop when validation loss stops improving
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),

        # Reduce learning rate when validation loss stops improving
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),

        # Save training history
        tf.keras.callbacks.CSVLogger(
            str(CSV_LOG_PATH)
        )
    ]

    print("Callbacks ready!", flush=True)

    return callbacks


# ==================================================
# 7. TRAIN DENSENET121
# ==================================================

def train_model(
    model,
    train_dataset,
    val_dataset,
    class_weights,
    callbacks
):

    print("\n======================================", flush=True)
    print("DENSENET121 TRAINING STARTED", flush=True)
    print("======================================", flush=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    return history


# ==================================================
# 8. SAVE FINAL WEIGHTS
# ==================================================

def save_final_weights(model):

    print("\n[STEP 5] Saving final weights...", flush=True)

    model.save_weights(str(FINAL_WEIGHTS_PATH))

    print(
        "Final weights saved:",
        FINAL_WEIGHTS_PATH,
        flush=True
    )


# ==================================================
# 9. DISPLAY TRAINING RESULTS
# ==================================================

def display_results(history):

    print("\n======================================", flush=True)
    print("DENSENET121 TRAINING RESULTS", flush=True)
    print("======================================", flush=True)

    val_losses = history.history["val_loss"]
    val_accuracies = history.history["val_accuracy"]

    # Find the epoch with the lowest validation loss
    best_epoch_index = int(np.argmin(val_losses))

    best_epoch = best_epoch_index + 1
    best_val_loss = float(val_losses[best_epoch_index])

    # Accuracy from the SAME best-loss epoch
    best_val_accuracy = float(
        val_accuracies[best_epoch_index]
    )

    print("Best epoch:", best_epoch, flush=True)

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}",
        flush=True
    )

    print(
        "Validation accuracy at best epoch:",
        f"{best_val_accuracy * 100:.2f}%",
        flush=True
    )

    print("\nSaved files:", flush=True)
    print("Best weights:", BEST_WEIGHTS_PATH, flush=True)
    print("Final weights:", FINAL_WEIGHTS_PATH, flush=True)
    print("Training log:", CSV_LOG_PATH, flush=True)


# ==================================================
# 10. MAIN EXECUTION
# ==================================================

def main():

    # Load dataset
    train_dataset, val_dataset, y_train = load_dataset()

    # Calculate class weights
    class_weights = calculate_class_weights(y_train)

    # Build DenseNet121
    model = load_model_architecture()

    # Prepare callbacks
    callbacks = create_callbacks()

    # Train model
    history = train_model(
        model,
        train_dataset,
        val_dataset,
        class_weights,
        callbacks
    )

    # EarlyStopping restores the best weights
    save_final_weights(model)

    # Display results
    display_results(history)

    print(
        "\nDENSENET121 TRAINING COMPLETED SUCCESSFULLY!",
        flush=True
    )


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\n======================================", flush=True)
        print("TRAINING FAILED", flush=True)
        print("======================================", flush=True)

        print("Error:", error, flush=True)

        traceback.print_exc()
