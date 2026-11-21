
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import preprocess_input

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_best.keras"

RESULTS_DIR = PROJECT_ROOT / "results" / "resnet50"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]

NUM_CLASSES = len(CLASS_NAMES)


# ==========================================
# 2. CPU CONFIGURATION
# ==========================================

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

print("\nEvaluation device: CPU")


# ==========================================
# 3. LOAD TEST DATASET
# ==========================================

print("\n[STEP 1] Loading test dataset...")

data_loader_path = PROJECT_ROOT / "src" / "02_data_loader.py"

spec = importlib.util.spec_from_file_location(
    "data_loader",
    data_loader_path
)

if spec is None or spec.loader is None:
    raise ImportError("Unable to load data loader.")

data_loader = importlib.util.module_from_spec(spec)

spec.loader.exec_module(data_loader)

test_dataset = data_loader.test_dataset

y_test = np.asarray(
    data_loader.y_test
).astype(int).reshape(-1)

print("Test samples:", len(y_test))


# ==========================================
# 4. LOAD RESNET50 MODEL
# ==========================================

print("\n[STEP 2] Loading ResNet50 model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    custom_objects={
        "preprocess_input": preprocess_input
    },
    compile=False,
    safe_mode=False
)

print("ResNet50 loaded successfully!")

print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ==========================================
# 5. GENERATE PREDICTIONS
# ==========================================

print("\n[STEP 3] Predicting test images...")

predictions = model.predict(
    test_dataset,
    verbose=1
)

if len(predictions) != len(y_test):
    raise ValueError(
        "Prediction count does not match test labels. "
        "Check that test_dataset is not shuffled."
    )

if predictions.shape[1] != NUM_CLASSES:
    raise ValueError(
        "Model output classes do not match CLASS_NAMES."
    )

y_pred = np.argmax(
    predictions,
    axis=1
)

print("Predictions completed!")


# ==========================================
# 6. TEST ACCURACY
# ==========================================

print("\n[STEP 4] Calculating test accuracy...")

test_accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print("RESNET50 TEST ACCURACY")
print("================================")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\n[STEP 5] Generating classification report...")

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(NUM_CLASSES),
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\n================================")
print("RESNET50 CLASSIFICATION REPORT")
print("================================")

print(report)

report_path = RESULTS_DIR / "classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("RESNET50 CLASSIFICATION REPORT\n\n")

    file.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n"
    )

    file.write(report)

print("Classification report saved:", report_path)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

print("\n[STEP 6] Generating confusion matrix...")

cm = confusion_matrix(
    y_test,
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
    "ResNet50 - Confusion Matrix",
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

confusion_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("Confusion matrix saved:", confusion_path)


# ==========================================
# 9. MULTI-CLASS ROC CURVE
# ==========================================

print("\n[STEP 7] Generating ROC curves...")

y_test_binary = label_binarize(
    y_test,
    classes=np.arange(NUM_CLASSES)
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

auc_scores = []

for i, class_name in enumerate(CLASS_NAMES):

    fpr, tpr, thresholds = roc_curve(
        y_test_binary[:, i],
        predictions[:, i]
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
    "ResNet50 - Multi-Class ROC Curve",
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
print("RESNET50 ROC-AUC RESULTS")
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
# 11. SAVE EVALUATION SUMMARY
# ==========================================

print("\n[STEP 8] Saving evaluation summary...")

summary_path = RESULTS_DIR / "evaluation_results.txt"

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("RESNET50 EVALUATION RESULTS\n")
    file.write("=" * 45 + "\n\n")

    file.write(
        f"Total Test Images: {len(y_test)}\n"
    )

    file.write(
        f"Number of Classes: {NUM_CLASSES}\n"
    )

    file.write(
        f"Test Accuracy: {test_accuracy * 100:.2f}%\n"
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
print("RESNET50 EVALUATION COMPLETED!")
print("================================")

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Macro ROC-AUC: {macro_auc:.4f}")

print("\nSaved files:")

print("Classification Report:", report_path)
print("Confusion Matrix:", confusion_path)
print("ROC Curve:", roc_path)
print("Evaluation Summary:", summary_path)

print("\nAll ResNet50 evaluation results saved successfully!")
