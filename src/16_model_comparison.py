
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. MODEL NAMES
# ==========================================

models = [
    "Custom CNN",
    "ResNet50",
    "DenseNet121"
]


# ==========================================
# 3. EXACT MODEL PERFORMANCE DATA
# ==========================================

metrics = {

    "Accuracy (%)": [
        91.81,
        91.81,
        87.09
    ],

    "Macro Precision": [
        0.9254,
        0.9258,
        0.8657555459585912
    ],

    "Macro Recall": [
        0.9315,
        0.9213,
        0.8906617123806138
    ],

    "Macro F1-score": [
        0.9283,
        0.9235,
        0.8774274027202557
    ],

    "Macro ROC-AUC": [
        0.9875,
        0.9871,
        0.9746225235222696
    ]
}


# ==========================================
# 4. CREATE COMPARISON DATAFRAME
# ==========================================

df = pd.DataFrame(
    metrics,
    index=models
)

print("\n======================================")
print("THREE MODEL PERFORMANCE COMPARISON")
print("======================================")

print(
    df.to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# ==========================================
# 5. SAVE COMPARISON TABLE
# ==========================================

csv_path = (
    RESULTS_DIR / "three_model_comparison.csv"
)

df.to_csv(
    csv_path,
    float_format="%.6f"
)

print("\nComparison table saved:", csv_path)


# ==========================================
# 6. ACCURACY COMPARISON
# ==========================================

print("\n[STEP 1] Creating accuracy comparison...")

plt.figure(figsize=(9, 6))

bars = plt.bar(
    models,
    metrics["Accuracy (%)"]
)

plt.title(
    "Test Accuracy Comparison",
    fontsize=14
)

plt.xlabel("Models")

plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.5,
        f"{height:.2f}%",
        ha="center"
    )

plt.tight_layout()

accuracy_path = (
    RESULTS_DIR / "model_accuracy_comparison.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ==========================================
# 7. PRECISION, RECALL AND F1 COMPARISON
# ==========================================

print(
    "\n[STEP 2] Creating classification metrics comparison..."
)

classification_metrics = [
    "Macro Precision",
    "Macro Recall",
    "Macro F1-score"
]

x = range(len(models))

width = 0.24

plt.figure(figsize=(11, 7))

for i, metric in enumerate(classification_metrics):

    values = metrics[metric]

    positions = [
        position + (i - 1) * width
        for position in x
    ]

    bars = plt.bar(
        positions,
        values,
        width,
        label=metric
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.005,
            f"{height:.3f}",
            ha="center",
            fontsize=8
        )

plt.title(
    "Precision, Recall and F1-score Comparison",
    fontsize=14
)

plt.xlabel("Models")

plt.ylabel("Score")

plt.xticks(
    list(x),
    models
)

plt.ylim(0, 1.08)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

classification_path = (
    RESULTS_DIR / "classification_metrics_comparison.png"
)

plt.savefig(
    classification_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ==========================================
# 8. ROC-AUC COMPARISON
# ==========================================

print("\n[STEP 3] Creating ROC-AUC comparison...")

plt.figure(figsize=(9, 6))

bars = plt.bar(
    models,
    metrics["Macro ROC-AUC"]
)

plt.title(
    "Macro ROC-AUC Comparison",
    fontsize=14
)

plt.xlabel("Models")

plt.ylabel("Macro ROC-AUC")

plt.ylim(0, 1.05)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.4f}",
        ha="center"
    )

plt.tight_layout()

roc_auc_path = (
    RESULTS_DIR / "model_roc_auc_comparison.png"
)

plt.savefig(
    roc_auc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ==========================================
# 9. PERFORMANCE RANKING
# ==========================================

print("\n[STEP 4] Calculating performance ranking...")

ranking = df.rank(
    ascending=False,
    method="min"
)

ranking_path = (
    RESULTS_DIR / "three_model_ranking.csv"
)

ranking.to_csv(
    ranking_path
)

print("\nMODEL RANKING")

print(
    ranking.to_string(
        float_format=lambda x: f"{x:.0f}"
    )
)


# ==========================================
# 10. BEST MODEL FOR EACH METRIC
# ==========================================

print("\n======================================")
print("BEST MODEL BY PERFORMANCE METRIC")
print("======================================")

best_models = {}

for metric in df.columns:

    best_model = df[metric].idxmax()

    best_score = df[metric].max()

    best_models[metric] = (
        best_model,
        best_score
    )

    print(
        f"{metric}: {best_model} "
        f"({best_score:.4f})"
    )


# ==========================================
# 11. SAVE FINAL COMPARISON REPORT
# ==========================================

print("\n[STEP 5] Saving final comparison report...")

report_path = (
    RESULTS_DIR / "final_model_comparison.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "THREE MODEL PERFORMANCE COMPARISON\n"
    )

    file.write(
        "==================================\n\n"
    )

    file.write(
        "Test dataset: 3,175 images\n"
    )

    file.write(
        "Number of classes: 4\n\n"
    )

    file.write(
        "PERFORMANCE METRICS\n"
    )

    file.write(
        df.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    file.write("\n\nMODEL RANKING\n")

    file.write(
        ranking.to_string(
            float_format=lambda x: f"{x:.0f}"
        )
    )

    file.write("\n\nBEST MODEL BY METRIC\n")

    for metric, result in best_models.items():

        best_model, best_score = result

        file.write(
            f"\n{metric}: {best_model} "
            f"({best_score:.4f})"
        )

    file.write(
        "\n\nEVALUATION NOTES\n"
    )

    file.write(
        "\nAll models were evaluated using the same "
        "3,175-image test dataset."
    )

    file.write(
        "\nDenseNet121 macro metrics use the exact "
        "values supplied in its evaluation report."
    )

    file.write(
        "\nThese results represent experimental model "
        "evaluation and do not establish clinical validation."
    )


# ==========================================
# 12. FINAL OUTPUT
# ==========================================

print("\n======================================")
print("MODEL COMPARISON COMPLETED!")
print("======================================")

print("\nGenerated files:")

print("Comparison CSV:", csv_path)

print("Accuracy chart:", accuracy_path)

print("Classification metrics chart:", classification_path)

print("ROC-AUC chart:", roc_auc_path)

print("Ranking CSV:", ranking_path)

print("Final comparison report:", report_path)

print("\nAll comparison results saved successfully!")
