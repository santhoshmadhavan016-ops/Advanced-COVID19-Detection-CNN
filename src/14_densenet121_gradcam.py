
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    pass

print("\nGrad-CAM device: CPU")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "models" /
    "densenet121_best_v2.weights.h5"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results" / "densenet121"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)


def load_module(name, path):

    spec = importlib.util.spec_from_file_location(
        name,
        path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


print("\n[STEP 1] Loading project modules...")

config = load_module(
    "project_config",
    PROJECT_ROOT / "src" / "01_config.py"
)

loader = load_module(
    "data_loader",
    PROJECT_ROOT / "src" / "02_data_loader.py"
)

architecture = load_module(
    "densenet_architecture",
    PROJECT_ROOT / "src" / "11_densenet121_architecture.py"
)

CLASS_NAMES = config.CLASS_NAMES

X_test = loader.X_test
y_test = np.asarray(loader.y_test)

print("Project modules loaded successfully!")
print("Test images:", len(X_test))


print("\n[STEP 2] Loading DenseNet121...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Best weights not found: {MODEL_PATH}"
    )

model = architecture.build_densenet121()

model.load_weights(str(MODEL_PATH))

print("DenseNet121 weights loaded successfully!")


print("\n[STEP 3] Building Grad-CAM model...")

base_model = model.get_layer("densenet121")

target_layer = base_model.get_layer(
    "conv5_block16_concat"
)

feature_extractor = tf.keras.Model(
    inputs=base_model.input,
    outputs=[
        target_layer.output,
        base_model.output
    ],
    name="densenet121_feature_extractor"
)

x = model.get_layer("rgb_conversion")(
    [
        model.input,
        model.input,
        model.input
    ]
)

x = model.get_layer("pixel_rescaling")(x)

x = model.get_layer("densenet_preprocessing")(x)

conv_outputs, backbone_outputs = feature_extractor(
    x,
    training=False
)

x = model.get_layer("global_average_pooling")(
    backbone_outputs
)

x = model.get_layer("dense_layer")(x)

x = model.get_layer("dropout")(
    x,
    training=False
)

predictions = model.get_layer("predictions")(x)

grad_model = tf.keras.Model(
    inputs=model.input,
    outputs=[
        conv_outputs,
        predictions
    ],
    name="DenseNet121_GradCAM"
)

print("Grad-CAM model created successfully!")


def load_test_image(image_index):

    image_path = str(X_test[image_index])

    image = tf.io.read_file(image_path)

    image = tf.io.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    input_image = tf.expand_dims(
        image,
        axis=0
    )

    return image, input_image, image_path


def generate_gradcam(input_image, class_index):

    with tf.GradientTape() as tape:

        conv_features, predictions = grad_model(
            input_image,
            training=False
        )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_features
    )

    if gradients is None:
        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_features = conv_features[0]

    heatmap = tf.reduce_sum(
        conv_features * pooled_gradients,
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

    return heatmap.numpy(), predictions


print("\n[STEP 4] Finding one test image for each class...")

class_indices = {}

for class_index, class_name in enumerate(CLASS_NAMES):

    indices = np.where(y_test == class_index)[0]

    if len(indices) == 0:
        print(f"{class_name}: No test images found")
        continue

    class_indices[class_index] = int(indices[0])

    print(
        f"{class_name}: Image index "
        f"{class_indices[class_index]}"
    )


print("\n[STEP 5] Generating Grad-CAM visualizations...")

for class_index, image_index in class_indices.items():

    class_name = CLASS_NAMES[class_index]

    print("\n--------------------------------------")
    print("Processing:", class_name)
    print("Image index:", image_index)
    print("--------------------------------------")

    actual_label = int(y_test[image_index])

    if actual_label < 0 or actual_label >= len(CLASS_NAMES):
        raise ValueError("Invalid test image class label.")

    image, input_image, image_path = load_test_image(
        image_index
    )

    heatmap, predictions = generate_gradcam(
        input_image,
        actual_label
    )

    predicted_class = int(
        tf.argmax(predictions[0]).numpy()
    )

    confidence = float(
        predictions[0, predicted_class].numpy()
    ) * 100

    actual_probability = float(
        predictions[0, actual_label].numpy()
    ) * 100

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        IMAGE_SIZE
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

    print("Actual class:", CLASS_NAMES[actual_label])
    print("Predicted class:", CLASS_NAMES[predicted_class])
    print(f"Predicted confidence: {confidence:.2f}%")
    print(f"Actual-class probability: {actual_probability:.2f}%")

    print("\nClass probabilities:")

    for i, name in enumerate(CLASS_NAMES):

        probability = float(
            predictions[0, i].numpy()
        ) * 100

        print(f"{name}: {probability:.2f}%")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5)
    )

    axes[0].imshow(
        original_image,
        cmap="gray"
    )

    axes[0].set_title("Original Chest X-ray")

    axes[1].imshow(
        heatmap_resized,
        cmap="jet"
    )

    axes[1].set_title("Grad-CAM Heatmap")

    axes[2].imshow(
        overlay
    )

    axes[2].set_title("Grad-CAM Overlay")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        f"Actual: {CLASS_NAMES[actual_label]} | "
        f"Predicted: {CLASS_NAMES[predicted_class]} "
        f"({confidence:.2f}%)",
        fontsize=13
    )

    plt.tight_layout()

    safe_class_name = (
        class_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    output_path = (
        RESULTS_DIR /
        f"gradcam_{safe_class_name}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("\nVisualization saved:")
    print(output_path)

    plt.show()

    plt.close(fig)


print("\n======================================")
print("ALL CLASS GRAD-CAM VISUALIZATIONS COMPLETED!")
print("======================================")
