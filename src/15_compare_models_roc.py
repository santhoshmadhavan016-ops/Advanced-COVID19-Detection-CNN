
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from tensorflow.keras.applications.resnet50 import preprocess_input


# ==========================================
# 1. CPU CONFIGURATION
# ==========================================

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

print("\nEvaluation device: CPU")


# ==========================================
# 2. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_CNN_PATH = MODELS_DIR / "custom_cnn_best.keras"

RESNET50_PATH = MODELS_DIR / "resnet50_best.keras"

DENSENET121_PATH = (
    MODELS_DIR / "densenet121_best_v2.weights.h5"
)

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]

NUM_CLASSES = len(CLASS_NAMES)


# ==========================================
# 3. DYNAMIC MODULE LOADER
# ==========================================

def load_module(module_name, file_path):

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Unable to load module: {file_path}"
        )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ==========================================
# 4. LOAD TEST DATASET
# ==========================================

print("\n[STEP 1] Loading test dataset...")

loader = load_module(
    "data_loader",
    SRC_DIR / "02_data_loader.py"
)

test_dataset = loader.test_dataset

y_test = np.asarray(
    loader.y_test
).astype(int).reshape(-1)

print("Test samples:", len(y_test))


# Verify dataset label order

dataset_labels = []

for images, labels in test_dataset:
    dataset_labels.extend(
        labels.numpy().reshape(-1)
    )

dataset_labels = np.asarray(
    dataset_labels
).astype(int)

if not np.array_equal(y_test, dataset_labels):
    raise ValueError(
        "Dataset labels do not match y_test order. "
        "Check test_dataset ordering before comparing models."
    )

print("Test dataset order verified!")


# ==========================================
# 5. LOAD CUSTOM CNN
# ==========================================

print("\n[STEP 2] Loading Custom CNN...")

if not CUSTOM_CNN_PATH.exists():
    raise FileNotFoundError(CUSTOM_CNN_PATH)

custom_cnn = tf.keras.models.load_model(
    str(CUSTOM_CNN_PATH),
    compile=False
)

print("Custom CNN loaded successfully!")


# ==========================================
# 6. LOAD RESNET50
# ==========================================

print("\n[STEP 3] Loading ResNet50...")

if not RESNET50_PATH.exists():
    raise FileNotFoundError(RESNET50_PATH)

resnet50 = tf.keras.models.load_model(
    str(RESNET50_PATH),
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False,
    safe_mode=False
)

print("ResNet50 loaded successfully!")


# ==========================================
# 7. LOAD DENSENET121
# ==========================================

print("\n[STEP 4] Building DenseNet121...")

architecture = load_module(
    "densenet_architecture",
    SRC_DIR / "11_densenet121_architecture.py"
)

densenet121 = architecture.build_densenet121()

if not DENSENET121_PATH.exists():
    raise FileNotFoundError(DENSENET121_PATH)

print("Loading DenseNet121 best weights...")

densenet121.load_weights(
    str(DENSENET121_PATH)
)

print("DenseNet121 loaded successfully!")


# ==========================================
# 8. GENERATE PREDICTIONS
# ==========================================

def generate_predictions(model, model_name):

    print(f"\nGenerating predictions: {model_name}")

    probabilities = model.predict(
        test_dataset,
        verbose=1
    )

    probabilities = np.asarray(probabilities)

    if probabilities.shape != (
        len(y_test),
        NUM_CLASSES
    ):
        raise ValueError(
            f"{model_name} prediction shape is incorrect: "
            f"{probabilities.shape}"
        )

    print(
        f"{model_name} predictions completed!"
    )

    return probabilities


print("\n[STEP 5] Generating predictions for all models...")

custom_predictions = generate_predictions(
    custom_cnn,
    "Custom CNN"
)

resnet_predictions = generate_predictions(
    resnet50,
    "ResNet50"
)

densenet_predictions = generate_predictions(
    densenet121,
    "DenseNet121"
)


# ==========================================
# 9. PREPARE MULTI-CLASS LABELS
# ==========================================

y_test_binary = label_binarize(
    y_test,
    classes=np.arange(NUM_CLASSES)
)


# ==========================================
# 10. CALCULATE MACRO ROC CURVE
# ==========================================

def calculate_macro_roc(y_true_binary, predictions):

    all_fpr = np.linspace(
        0,
        1,
        1001
    )

    mean_tpr = np.zeros_like(all_fpr)

    class_auc_scores = []

    for i in range(NUM_CLASSES):

        fpr, tpr, thresholds = roc_curve(
            y_true_binary[:, i],
            predictions[:, i]
        )

        class_auc = auc(
            fpr,
            tpr
        )

        class_auc_scores.append(class_auc)

        mean_tpr += np.interp(
            all_fpr,
            fpr,
            tpr
        )

    mean_tpr /= NUM_CLASSES

    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    macro_auc = float(
        np.mean(class_auc_scores)
    )

    return all_fpr, mean_tpr, macro_auc, class_auc_scores


# ==========================================
# 11. CALCULATE ROC FOR ALL MODELS
# ==========================================

print("\n[STEP 6] Calculating macro ROC curves...")

custom_fpr, custom_tpr, custom_auc, custom_class_auc = (
    calculate_macro_roc(
        y_test_binary,
        custom_predictions
    )
)

resnet_fpr, resnet_tpr, resnet_auc, resnet_class_auc = (
    calculate_macro_roc(
        y_test_binary,
        resnet_predictions
    )
)

densenet_fpr, densenet_tpr, densenet_auc, densenet_class_auc = (
    calculate_macro_roc(
        y_test_binary,
        densenet_predictions
    )
)


# ==========================================
# 12. PLOT COMBINED ROC CURVE
# ==========================================

print("\n[STEP 7] Creating combined ROC curve...")

fig, ax = plt.subplots(
    figsize=(10, 8)
)

ax.plot(
    custom_fpr,
    custom_tpr,
    linewidth=2.5,
    label=f"Custom CNN (Macro AUC = {custom_auc:.4f})"
)

ax.plot(
    resnet_fpr,
    resnet_tpr,
    linewidth=2.5,
    label=f"ResNet50 (Macro AUC = {resnet_auc:.4f})"
)

ax.plot(
    densenet_fpr,
    densenet_tpr,
    linewidth=2.5,
    label=f"DenseNet121 (Macro AUC = {densenet_auc:.4f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier"
)

ax.set_title(
    "Combined Macro-Average ROC Curve",
    fontsize=15
)

ax.set_xlabel(
    "False Positive Rate",
    fontsize=12
)

ax.set_ylabel(
    "True Positive Rate",
    fontsize=12
)

ax.legend(
    loc="lower right"
)

ax.grid(alpha=0.3)

plt.tight_layout()

combined_roc_path = (
    RESULTS_DIR / "combined_three_model_roc_curve.png"
)

plt.savefig(
    combined_roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close(fig)

print("Combined ROC curve saved:", combined_roc_path)


# ==========================================
# 13. PRINT CLASS-WISE AUC RESULTS
# ==========================================

print("\n==========================================")
print("THREE MODEL CLASS-WISE ROC-AUC COMPARISON")
print("==========================================")

for i, class_name in enumerate(CLASS_NAMES):

    print(f"\n{class_name}")

    print(
        f"Custom CNN: {custom_class_auc[i]:.4f}"
    )

    print(
        f"ResNet50: {resnet_class_auc[i]:.4f}"
    )

    print(
        f"DenseNet121: {densenet_class_auc[i]:.4f}"
    )


# ==========================================
# 14. FINAL MACRO AUC COMPARISON
# ==========================================

print("\n==========================================")
print("FINAL MACRO ROC-AUC COMPARISON")
print("==========================================")

print(f"Custom CNN: {custom_auc:.4f}")

print(f"ResNet50: {resnet_auc:.4f}")

print(f"DenseNet121: {densenet_auc:.4f}")


# ==========================================
# 15. SAVE COMPARISON RESULTS
# ==========================================

comparison_path = (
    RESULTS_DIR / "three_model_roc_auc_comparison.txt"
)

with open(
    comparison_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("THREE MODEL ROC-AUC COMPARISON\n")
    file.write("==============================\n\n")

    file.write(
        f"Test samples: {len(y_test)}\n\n"
    )

    file.write("CLASS-WISE ROC-AUC\n")

    for i, class_name in enumerate(CLASS_NAMES):

        file.write(f"\n{class_name}\n")

        file.write(
            f"Custom CNN: {custom_class_auc[i]:.4f}\n"
        )

        file.write(
            f"ResNet50: {resnet_class_auc[i]:.4f}\n"
        )

        file.write(
            f"DenseNet121: {densenet_class_auc[i]:.4f}\n"
        )

    file.write("\nMACRO ROC-AUC\n")

    file.write(
        f"Custom CNN: {custom_auc:.4f}\n"
    )

    file.write(
        f"ResNet50: {resnet_auc:.4f}\n"
    )

    file.write(
        f"DenseNet121: {densenet_auc:.4f}\n"
    )


# ==========================================
# 16. COMPLETION
# ==========================================

print("\n==========================================")
print("THREE MODEL ROC COMPARISON COMPLETED!")
print("==========================================")

print("\nSaved files:")

print("Combined ROC:", combined_roc_path)

print("AUC Comparison:", comparison_path)

print("\nAll three models evaluated on the same test dataset.")
