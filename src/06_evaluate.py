
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "custom_cnn_best.keras"

RESULTS_DIR = PROJECT_ROOT / "results" / "custom_cnn"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]

NUM_CLASSES = len(CLASS_NAMES)


# ==========================================
# 3. LOAD TEST DATASET
# ==========================================

print("\n[STEP 1] Loading test dataset...")

SRC_DIR = Path(__file__).resolve().parent

loader_path = SRC_DIR / "02_data_loader.py"

spec = importlib.util.spec_from_file_location(
    "data_loader",
    loader_path
)

if spec is None or spec.loader is None:
    raise ImportError("Unable to load data loader.")

data_loader = importlib.util.module_from_spec(spec)

spec.loader.exec_module(data_loader)

test_dataset = data_loader.test_dataset

print("Test dataset loaded successfully!")


# ==========================================
# 4. LOAD TRAINED CUSTOM CNN MODEL
# ==========================================

print("\n[STEP 2] Loading Custom CNN model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Custom CNN model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Custom CNN loaded successfully!")

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ==========================================
# 5. GENERATE MODEL PREDICTIONS
# ==========================================

print("\n[STEP 3] Generating predictions...")

y_true = []
y_prob = []

for images, labels in test_dataset:

    predictions = model(
        images,
        training=False
    ).numpy()

    y_prob.extend(predictions)

    y_true.extend(
        labels.numpy().reshape(-1)
    )


y_true = np.asarray(y_true).astype(int)
y_prob = np.asarray(y_prob)

if y_prob.shape[1] != NUM_CLASSES:
    raise ValueError(
        "Model output classes do not match CLASS_NAMES."
    )

if len(y_true) != len(y_prob):
    raise ValueError(
        "Prediction count does not match test labels."
    )

y_pred = np.argmax(
    y_prob,
    axis=1
)

print("Predictions completed!")
print("Total test images:", len(y_true))


# ==========================================
# 6. TEST ACCURACY
# ==========================================

print("\n[STEP 4] Calculating test accuracy...")

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n================================")
print("CUSTOM CNN TEST ACCURACY")
print("================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n[STEP 5] Generating classification report...")

report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n================================")
print("CUSTOM CNN CLASSIFICATION REPORT")
print("================================")

print(report)

report_path = RESULTS_DIR / "classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("CUSTOM CNN CLASSIFICATION REPORT\n\n")

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    file.write(report)

print("Classification report saved:", report_path)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

print("\n[STEP 6] Generating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(NUM_CLASSES)
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    cmap="Blues",
    values_format="d",
    ax=ax,
    colorbar=False
)

ax.set_title(
    "Custom CNN - Confusion Matrix",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel(
    "Predicted Label",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "True Label",
    fontsize=12,
    fontweight="bold"
)

plt.xticks(rotation=20, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

cm_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Confusion matrix saved:", cm_path)


# ==========================================
# 9. MULTI-CLASS ROC CURVE
# ==========================================

print("\n[STEP 7] Generating ROC curves...")

y_true_binary = label_binarize(
    y_true,
    classes=np.arange(NUM_CLASSES)
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

auc_scores = []

for i, class_name in enumerate(CLASS_NAMES):

    fpr, tpr, thresholds = roc_curve(
        y_true_binary[:, i],
        y_prob[:, i]
    )

    class_auc = auc(
        fpr,
        tpr
    )

    auc_scores.append(class_auc)

    ax.plot(
        fpr,
        tpr,
        linewidth=2.5,
        label=(
            f"{class_name} "
            f"(AUC = {class_auc:.4f})"
        )
    )


ax.plot(
    [0, 1],
    [0, 1],
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier"
)

ax.set_title(
    "Custom CNN - Multi-Class ROC Curve",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel(
    "False Positive Rate",
    fontsize=12,
    fontweight="bold"
)

ax.set_ylabel(
    "True Positive Rate",
    fontsize=12,
    fontweight="bold"
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)

ax.legend(
    loc="lower right",
    fontsize=10,
    frameon=True
)

ax.grid(
    alpha=0.3,
    linestyle="--"
)

plt.tight_layout()

roc_path = RESULTS_DIR / "roc_curve.png"

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("ROC curve saved:", roc_path)


# ==========================================
# 10. ROC-AUC RESULTS
# ==========================================

macro_auc = float(
    np.mean(auc_scores)
)

print("\n================================")
print("CUSTOM CNN ROC-AUC RESULTS")
print("================================")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name}: "
        f"{auc_scores[i]:.4f}"
    )

print(
    f"\nMacro ROC-AUC: {macro_auc:.4f}"
)


# ==========================================
# 11. SAVE COMPLETE EVALUATION SUMMARY
# ==========================================

print("\n[STEP 8] Saving evaluation summary...")

summary_path = RESULTS_DIR / "evaluation_results.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("CUSTOM CNN EVALUATION RESULTS\n")
    file.write("=" * 45 + "\n\n")

    file.write(
        f"Total Test Images: {len(y_true)}\n"
    )

    file.write(
        f"Number of Classes: {NUM_CLASSES}\n"
    )

    file.write(
        f"Test Accuracy: {accuracy * 100:.2f}%\n"
    )

    file.write(
        f"Macro ROC-AUC: {macro_auc:.4f}\n\n"
    )

    file.write("CLASSIFICATION REPORT\n")
    file.write("-" * 45 + "\n")

    file.write(report)

    file.write("\n\nCLASS-WISE ROC-AUC RESULTS\n")
    file.write("-" * 45 + "\n")

    for i, class_name in enumerate(CLASS_NAMES):

        file.write(
            f"{class_name}: {auc_scores[i]:.4f}\n"
        )

    file.write(
        f"\nMacro ROC-AUC: {macro_auc:.4f}\n"
    )


# ==========================================
# 12. FINAL RESULTS
# ==========================================

print("\n================================")
print("CUSTOM CNN EVALUATION COMPLETED!")
print("================================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Macro ROC-AUC: {macro_auc:.4f}")

print("\nSaved files:")

print("Classification Report:", report_path)
print("Confusion Matrix:", cm_path)
print("ROC Curve:", roc_path)
print("Evaluation Summary:", summary_path)

print("\nAll Custom CNN evaluation results saved successfully!")
