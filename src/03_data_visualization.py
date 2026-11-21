
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Load dataset split
split_file = np.load(
    DATA_DIR / "dataset_splits.npz",
    allow_pickle=False
)

X_train = split_file["X_train"]
y_train = split_file["y_train"]

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]


# Display one image from each class
plt.figure(figsize=(12, 5))

for class_id, class_name in enumerate(CLASS_NAMES):

    # Find first image belonging to this class
    index = np.where(y_train == class_id)[0][0]

    image_path = DATA_DIR / X_train[index]

    # Read grayscale image
    image = plt.imread(image_path)

    plt.subplot(1, 4, class_id + 1)
    plt.imshow(image, cmap="gray")
    plt.title(class_name)
    plt.axis("off")


plt.suptitle("COVID-19 Dataset - Sample X-ray Images")
plt.tight_layout()
plt.show()
