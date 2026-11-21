
import numpy as np
import tensorflow as tf
from pathlib import Path

# Import configuration
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
RANDOM_SEED = 42

DATA_DIR = PROJECT_ROOT / "data"
SPLIT_FILE = DATA_DIR / "dataset_splits.npz"


# Load saved dataset splits
data = np.load(SPLIT_FILE, allow_pickle=False)


def resolve_image_path(image_path):
    image_path = str(image_path).replace("\\", "/")
    return str(DATA_DIR / image_path)


# Resolve image paths
X_train = np.array([resolve_image_path(p) for p in data["X_train"]])
X_val = np.array([resolve_image_path(p) for p in data["X_val"]])
X_test = np.array([resolve_image_path(p) for p in data["X_test"]])

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]


# Load and preprocess images
def load_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# Create TensorFlow datasets
def create_dataset(paths, labels, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=RANDOM_SEED
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_dataset = create_dataset(X_train, y_train, training=True)
val_dataset = create_dataset(X_val, y_val)
test_dataset = create_dataset(X_test, y_test)


if __name__ == "__main__":

    print("Training batches:", len(train_dataset))
    print("Validation batches:", len(val_dataset))
    print("Testing batches:", len(test_dataset))

    for images, labels in train_dataset.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
