
import importlib.util
from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt


# ==========================================
# 1. CPU CONFIGURATION
# ==========================================

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

print("\nGrad-CAM device: CPU")


# ==========================================
# 2. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results" / "resnet50"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_best.keras"

OUTPUT_PATH = RESULTS_DIR / "gradcam_visualization.png"


# ==========================================
# 3. CLASS NAMES
# ==========================================

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]


# ==========================================
# 4. LOAD TEST DATASET
# ==========================================

print("\n[STEP 1] Loading test dataset...")

loader_path = PROJECT_ROOT / "src" / "02_data_loader.py"

spec = importlib.util.spec_from_file_location(
    "data_loader",
    loader_path
)

data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)

test_dataset = data_loader.test_dataset

print("Test dataset loaded successfully!")


# ==========================================
# 5. LOAD RESNET50 MODEL
# ==========================================

print("\n[STEP 2] Loading ResNet50 model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "preprocess_input":
            tf.keras.applications.resnet50.preprocess_input
    },
    compile=False,
    safe_mode=False
)

print("ResNet50 model loaded successfully!")
print("Model input:", model.input_shape)
print("Model output:", model.output_shape)


# ==========================================
# 6. FIND LAST CONVOLUTIONAL LAYER
# ==========================================

print("\n[STEP 3] Finding last convolutional layer...")


def find_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.Model):

            for sub_layer in reversed(layer.layers):

                if isinstance(
                    sub_layer,
                    tf.keras.layers.Conv2D
                ):
                    return layer, sub_layer.name

    raise ValueError(
        "Could not find a convolutional layer."
    )


base_model, last_conv_name = find_last_conv_layer(model)

print("Target convolutional layer:", last_conv_name)


# ==========================================
# 7. BUILD FEATURE EXTRACTION MODEL
# ==========================================

print("\n[STEP 4] Building connected Grad-CAM model...")

last_conv_layer = base_model.get_layer(last_conv_name)

base_index = model.layers.index(base_model)

feature_model = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        last_conv_layer.output,
        base_model.output
    ]
)

print("Feature extraction model created successfully!")


# ==========================================
# 8. GRAD-CAM FUNCTION
# ==========================================

def generate_gradcam(image):

    image = tf.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:

        # Follow the original model layers
        # before the ResNet50 backbone.

        x = image

        for layer in model.layers[1:base_index]:

            if isinstance(
                layer,
                tf.keras.layers.Concatenate
            ):
                x = layer([x, x, x])

            else:
                x = layer(x)

        # Extract convolutional features and
        # backbone output in the same forward pass.

        conv_outputs, backbone_features = feature_model(
            x,
            training=False
        )

        # Continue through the original classifier head.

        x = backbone_features

        for layer in model.layers[base_index + 1:]:

            if isinstance(
                layer,
                tf.keras.layers.Dropout
            ):
                x = layer(x, training=False)

            else:
                x = layer(x)

        predictions = x

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_score = predictions[:, predicted_class]

    # Calculate gradients.

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Gradients are still unavailable. "
            "Check the ResNet50 computation graph."
        )

    # Calculate channel importance.

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # Generate heatmap.

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224)
    )

    heatmap = tf.squeeze(heatmap).numpy()

    confidence = float(
        tf.reduce_max(predictions[0]).numpy()
    )

    return (
        heatmap,
        int(predicted_class.numpy()),
        confidence
    )


# ==========================================
# 9. SELECT TEST IMAGE
# ==========================================

print("\n[STEP 5] Selecting test image...")

selected_image = None
actual_label = None

for images, labels in test_dataset.take(1):

    selected_image = images[0]
    actual_label = int(labels[0].numpy())

if selected_image is None:
    raise ValueError("No test image found.")

print("Test image selected successfully!")


# ==========================================
# 10. GENERATE GRAD-CAM
# ==========================================

print("\n[STEP 6] Generating Grad-CAM...")

heatmap, predicted_class, confidence = generate_gradcam(
    selected_image
)

actual_name = CLASS_NAMES[actual_label]
predicted_name = CLASS_NAMES[predicted_class]

print("\n================================")
print("RESNET50 GRAD-CAM RESULTS")
print("================================")

print("Actual class:", actual_name)
print("Predicted class:", predicted_name)
print(f"Confidence: {confidence * 100:.2f}%")


# ==========================================
# 11. CREATE VISUALIZATION
# ==========================================

print("\n[STEP 7] Creating visualization...")

original_image = selected_image.numpy().squeeze()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)

plt.imshow(
    original_image,
    cmap="gray"
)

plt.title(
    "Original Chest X-ray",
    fontsize=13,
    fontweight="bold"
)

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(
    heatmap,
    cmap="jet"
)

plt.title(
    "Grad-CAM Heatmap",
    fontsize=13,
    fontweight="bold"
)

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(
    original_image,
    cmap="gray"
)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.4
)

plt.title(
    "ResNet50 Grad-CAM Overlay",
    fontsize=13,
    fontweight="bold"
)

plt.axis("off")


plt.suptitle(
    f"ResNet50 Explainable AI\n"
    f"Actual: {actual_name} | "
    f"Predicted: {predicted_name} | "
    f"Confidence: {confidence * 100:.2f}%",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 12. COMPLETION
# ==========================================

print("\n================================")
print("RESNET50 GRAD-CAM COMPLETED!")
print("================================")

print("Last convolutional layer:", last_conv_name)
print("Actual class:", actual_name)
print("Predicted class:", predicted_name)
print(f"Confidence: {confidence * 100:.2f}%")

print("\nGrad-CAM saved successfully:")
print(OUTPUT_PATH)
