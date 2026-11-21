
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


# ==================================================
# 1. CPU CONFIGURATION
# ==================================================

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

print("\nEvaluation device: CPU")


# ==================================================
# 2. PROJECT PATHS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "models" /
    "densenet121_best_v2.weights.h5"
)

RESULT_DIR = PROJECT_ROOT / "results" / "densenet121"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

CONFUSION_PATH = RESULT_DIR / "confusion_matrix.png"
ROC_PATH = RESULT_DIR / "roc_curve.png"
REPORT_PATH = RESULT_DIR / "classification_report.csv"
RESULTS_PATH = RESULT_DIR / "evaluation_results.txt"


# ==================================================
# 3. CONFIGURATION
# ==================================================

NUM_CLASSES = 4

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]


# ==================================================
# 4. LOAD PYTHON MODULES
# ==================================================

def load_module(module_name, file_path):

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required Python file not found: {file_path}"
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        file_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not load module: {file_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ==================================================
# 5. LOAD TEST DATASET
# ==================================================

def load_test_data():

    print("\n[STEP 1] Loading test dataset...")

    loader_path = PROJECT_ROOT / "src" / "02_data_loader.py"

    loader = load_module("data_loader", loader_path)

    test_dataset = loader.test_dataset
    y_test = np.asarray(loader.y_test).astype(int)

    print("Test samples:", len(y_test))
    print("Test batches:", len(test_dataset))
    print("Test labels:", np.unique(y_test))

    if not np.array_equal(
        np.unique(y_test),
        np.arange(NUM_CLASSES)
    ):
        raise ValueError(
            "Test dataset does not contain all four expected labels."
        )

    return test_dataset, y_test


# ==================================================
# 6. LOAD TRAINED MODEL
# ==================================================

def load_trained_model():

    print("\n[STEP 2] Building DenseNet121...")

    architecture_path = (
        PROJECT_ROOT / "src" /
        "11_densenet121_architecture.py"
    )

    architecture = load_module(
        "densenet_architecture",
        architecture_path
    )

    model = architecture.build_densenet121()

    print("\nLoading best model weights...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    model.load_weights(str(MODEL_PATH))

    print("Best weights loaded successfully!")

    return model


# ==================================================
# 7. GENERATE PREDICTIONS
# ==================================================

def generate_predictions(model, test_dataset):

    print("\n[STEP 3] Generating predictions...")

    predictions = model.predict(
        test_dataset,
        verbose=1
    )

    y_pred = np.argmax(predictions, axis=1)

    print("Predictions completed!")
    print("Prediction shape:", predictions.shape)

    return predictions, y_pred


# ==================================================
# 8. CLASSIFICATION METRICS
# ==================================================

def calculate_metrics(y_true, y_pred, predictions):

    print("\n[STEP 4] Calculating evaluation metrics...")

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(NUM_CLASSES)
    )

    print("\n======================================")
    print("DENSENET121 CLASSIFICATION REPORT")
    print("======================================")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=np.arange(NUM_CLASSES),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("Overall Accuracy:", f"{accuracy * 100:.2f}%")

    return accuracy, report, matrix


# ==================================================
# 9. SENSITIVITY AND SPECIFICITY
# ==================================================

def calculate_sensitivity_specificity(matrix):

    print("\n[STEP 5] Calculating sensitivity and specificity...")

    total = np.sum(matrix)

    results = []

    for i, class_name in enumerate(CLASS_NAMES):

        tp = matrix[i, i]
        fn = np.sum(matrix[i, :]) - tp
        fp = np.sum(matrix[:, i]) - tp
        tn = total - tp - fn - fp

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0 else 0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0 else 0
        )

        results.append({
            "Class": class_name,
            "Sensitivity": sensitivity,
            "Specificity": specificity
        })

        print(f"\n{class_name}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

    return results


# ==================================================
# 10. CONFUSION MATRIX
# ==================================================

def save_confusion_matrix(matrix):

    print("\n[STEP 6] Saving confusion matrix...")

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(matrix, cmap="Blues")

    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="DenseNet121 Confusion Matrix"
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right"
    )

    threshold = matrix.max() / 2 if matrix.max() else 0

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):

            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black"
            )

    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=300)
    plt.close(fig)

    print("Confusion matrix saved:", CONFUSION_PATH)


# ==================================================
# 11. ROC CURVE AND AUC
# ==================================================

def save_roc_curve(y_true, predictions):

    print("\n[STEP 7] Generating ROC curves...")

    y_true_binary = label_binarize(
        y_true,
        classes=np.arange(NUM_CLASSES)
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    auc_values = {}

    for i, class_name in enumerate(CLASS_NAMES):

        fpr, tpr, _ = roc_curve(
            y_true_binary[:, i],
            predictions[:, i]
        )

        class_auc = auc(fpr, tpr)
        auc_values[class_name] = class_auc

        ax.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC = {class_auc:.4f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="DenseNet121 Multiclass ROC Curve"
    )

    ax.legend(loc="lower right")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(ROC_PATH, dpi=300)
    plt.close(fig)

    macro_auc = roc_auc_score(
        y_true_binary,
        predictions,
        average="macro"
    )

    print("\nClass-wise AUC:")

    for name, value in auc_values.items():
        print(f"{name}: {value:.4f}")

    print("Macro ROC-AUC:", f"{macro_auc:.4f}")
    print("ROC curve saved:", ROC_PATH)

    return auc_values, macro_auc


# ==================================================
# 12. SAVE EVALUATION RESULTS
# ==================================================

def save_results(
    accuracy,
    report,
    sensitivity_results,
    auc_values,
    macro_auc
):

    print("\n[STEP 8] Saving evaluation results...")

    rows = []

    for class_name in CLASS_NAMES:

        rows.append({
            "Class": class_name,
            "Precision": report[class_name]["precision"],
            "Recall": report[class_name]["recall"],
            "F1-score": report[class_name]["f1-score"],
            "Support": report[class_name]["support"],
            "Sensitivity": next(
                item["Sensitivity"]
                for item in sensitivity_results
                if item["Class"] == class_name
            ),
            "Specificity": next(
                item["Specificity"]
                for item in sensitivity_results
                if item["Class"] == class_name
            ),
            "ROC-AUC": auc_values[class_name]
        })

    rows.append({
        "Class": "Macro Average",
        "Precision": report["macro avg"]["precision"],
        "Recall": report["macro avg"]["recall"],
        "F1-score": report["macro avg"]["f1-score"],
        "Support": report["macro avg"]["support"],
        "Sensitivity": np.mean([
            item["Sensitivity"] for item in sensitivity_results
        ]),
        "Specificity": np.mean([
            item["Specificity"] for item in sensitivity_results
        ]),
        "ROC-AUC": macro_auc
    })

    pd.DataFrame(rows).to_csv(
        REPORT_PATH,
        index=False
    )

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:

        file.write("DENSENET121 MODEL EVALUATION\n")
        file.write("============================\n\n")

        file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"Macro ROC-AUC: {macro_auc:.4f}\n\n")

        file.write("CLASS-WISE ROC-AUC\n")

        for name, value in auc_values.items():
            file.write(f"{name}: {value:.4f}\n")

        file.write("\nCLASSIFICATION REPORT\n")

        for name in CLASS_NAMES:
            file.write(f"\n{name}\n")

            for metric in [
                "precision",
                "recall",
                "f1-score",
                "support"
            ]:
                file.write(
                    f"{metric}: {report[name][metric]}\n"
                )

        file.write("\nSENSITIVITY AND SPECIFICITY\n")

        for item in sensitivity_results:
            file.write(
                f"\n{item['Class']}\n"
                f"Sensitivity: {item['Sensitivity']:.4f}\n"
                f"Specificity: {item['Specificity']:.4f}\n"
            )

    print("Classification report saved:", REPORT_PATH)
    print("Evaluation results saved:", RESULTS_PATH)


# ==================================================
# 13. MAIN EXECUTION
# ==================================================

def main():

    print("\n======================================")
    print("DENSENET121 MODEL EVALUATION")
    print("======================================")

    test_dataset, y_true = load_test_data()

    model = load_trained_model()

    predictions, y_pred = generate_predictions(
        model,
        test_dataset
    )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Prediction and test label counts do not match."
        )

    if predictions.shape[1] != NUM_CLASSES:
        raise ValueError(
            "Model output does not contain four class probabilities."
        )

    accuracy, report, matrix = calculate_metrics(
        y_true,
        y_pred,
        predictions
    )

    sensitivity_results = calculate_sensitivity_specificity(
        matrix
    )

    save_confusion_matrix(matrix)

    auc_values, macro_auc = save_roc_curve(
        y_true,
        predictions
    )

    save_results(
        accuracy,
        report,
        sensitivity_results,
        auc_values,
        macro_auc
    )

    print("\n======================================")
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("======================================")

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro ROC-AUC: {macro_auc:.4f}")
    print("\nResults folder:", RESULT_DIR)


if __name__ == "__main__":
    main()
