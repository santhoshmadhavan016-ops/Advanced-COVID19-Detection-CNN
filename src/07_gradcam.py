
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras import layers


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "custom_cnn_best.keras"
DATA_PATH = PROJECT_ROOT / "data" / "dataset_splits.npz"

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


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

print("\n[STEP 1] Loading Custom CNN model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ==========================================
# 4. FIND LAST CONVOLUTIONAL LAYER
# ==========================================

print("\n[STEP 2] Finding last convolutional layer...")

last_conv_layer = None

for layer in reversed(model.layers):

    if isinstance(layer, layers.Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise ValueError(
        "No Conv2D layer found in the model."
    )

print("Last convolutional layer:", last_conv_layer.name)


# ==========================================
# 5. LOAD TEST IMAGE
# ==========================================

print("\n[STEP 3] Loading test image...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset split file not found: {DATA_PATH}"
    )

data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"]

image_path = X_test[0]

if isinstance(image_path, bytes):
    image_path = image_path.decode("utf-8")

image_path = Path(image_path)

if not image_path.is_absolute():
    image_path = PROJECT_ROOT / "data" / image_path

if not image_path.exists():
    raise FileNotFoundError(
        f"Test image not found: {image_path}"
    )

actual_label = int(y_test[0])

if actual_label < 0 or actual_label >= len(CLASS_NAMES):
    raise ValueError("Invalid actual class label.")

image = tf.io.read_file(str(image_path))

image = tf.image.decode_png(
    image,
    channels=1
)

image = tf.image.resize(
    image,
    (224, 224)
)

image = tf.cast(image, tf.float32) / 255.0

input_image = tf.expand_dims(
    image,
    axis=0
)

print("Test image loaded successfully!")
print("Actual class:", CLASS_NAMES[actual_label])


# ==========================================
# 6. CREATE GRAD-CAM MODEL
# ==========================================

print("\n[STEP 4] Creating Grad-CAM model...")

grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[
        last_conv_layer.output,
        model.output
    ]
)


# ==========================================
# 7. GENERATE GRAD-CAM HEATMAP
# ==========================================

print("\n[STEP 5] Generating Grad-CAM heatmap...")

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        input_image,
        training=False
    )

    predicted_class_tensor = tf.argmax(
        predictions[0]
    )

    class_score = predictions[:, predicted_class_tensor]


grads = tape.gradient(
    class_score,
    conv_outputs
)

if grads is None:
    raise ValueError(
        "Could not calculate Grad-CAM gradients."
    )

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)

conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(
    conv_outputs * pooled_grads,
    axis=-1
)

heatmap = tf.maximum(
    heatmap,
    0
)

heatmap = heatmap / (
    tf.reduce_max(heatmap) +
    tf.keras.backend.epsilon()
)

heatmap = heatmap.numpy()


# ==========================================
# 8. RESIZE AND COLOR HEATMAP
# ==========================================

print("\n[STEP 6] Creating heatmap overlay...")

heatmap_resized = tf.image.resize(
    heatmap[..., np.newaxis],
    (224, 224)
).numpy().squeeze()

heatmap_colored = plt.get_cmap("jet")(
    heatmap_resized
)[..., :3]

original_image = image.numpy().squeeze()

original_rgb = np.stack(
    [original_image] * 3,
    axis=-1
)

overlay = (
    0.6 * original_rgb +
    0.4 * heatmap_colored
)

overlay = np.clip(
    overlay,
    0,
    1
)


# ==========================================
# 9. PREDICTION RESULTS
# ==========================================

predicted_class = int(
    predicted_class_tensor.numpy()
)

confidence = (
    float(predictions[0, predicted_class].numpy())
    * 100
)

print("\n================================")
print("CUSTOM CNN GRAD-CAM RESULTS")
print("================================")

print("Actual Class:", CLASS_NAMES[actual_label])

print("Predicted Class:", CLASS_NAMES[predicted_class])

print(f"Confidence: {confidence:.2f}%")


# ==========================================
# 10. CREATE GRAD-CAM VISUALIZATION
# ==========================================

print("\n[STEP 7] Creating Grad-CAM visualization...")

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

axes[0].imshow(
    original_image,
    cmap="gray"
)

axes[0].set_title(
    "Original X-ray",
    fontsize=13,
    fontweight="bold"
)

axes[1].imshow(
    heatmap_resized,
    cmap="jet"
)

axes[1].set_title(
    "Grad-CAM Heatmap",
    fontsize=13,
    fontweight="bold"
)

axes[2].imshow(overlay)

axes[2].set_title(
    "Grad-CAM Overlay",
    fontsize=13,
    fontweight="bold"
)

for ax in axes:
    ax.axis("off")

fig.suptitle(
    f"Custom CNN Grad-CAM | "
    f"Actual: {CLASS_NAMES[actual_label]} | "
    f"Predicted: {CLASS_NAMES[predicted_class]} "
    f"({confidence:.2f}%)",
    fontsize=14,
    fontweight="bold",
    y=1.03
)

plt.tight_layout()


# ==========================================
# 11. SAVE GRAD-CAM RESULTS
# ==========================================

print("\n[STEP 8] Saving Grad-CAM visualization...")

output_path = (
    RESULTS_DIR / "gradcam_visualization.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

plt.close(fig)

print("\n================================")
print("CUSTOM CNN GRAD-CAM COMPLETED!")
print("================================")

print("Actual Class:", CLASS_NAMES[actual_label])
print("Predicted Class:", CLASS_NAMES[predicted_class])
print(f"Confidence: {confidence:.2f}%")

print("\nGrad-CAM visualization saved:")
print(output_path)

print("\nGrad-CAM completed successfully!")
