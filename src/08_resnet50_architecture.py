
import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

IMAGE_SIZE = (224, 224)
NUM_CLASSES = 4

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]


# ==========================================
# 2. LOAD PRETRAINED RESNET50
# ==========================================

print("\nLoading pretrained ResNet50...")

base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

print("ResNet50 loaded successfully!")


# ==========================================
# 3. FREEZE PRETRAINED LAYERS
# ==========================================

base_model.trainable = False

print("Pretrained ResNet50 layers frozen.")


# ==========================================
# 4. BUILD OUR CLASSIFICATION MODEL
# ==========================================

inputs = layers.Input(
    shape=(224, 224, 1),
    name="grayscale_input"
)

# Convert grayscale into three identical RGB channels
x = layers.Concatenate(axis=-1)([
    inputs,
    inputs,
    inputs
])

# Convert pixel values from 0-1 to 0-255
x = layers.Rescaling(255.0)(x)

# Apply ResNet50 preprocessing
x = layers.Lambda(
    tf.keras.applications.resnet50.preprocess_input,
    name="resnet_preprocessing"
)(x)

# Extract features using pretrained ResNet50
x = base_model(x, training=False)

# Classification layers
x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.5)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="predictions"
)(x)


# ==========================================
# 5. CREATE FINAL MODEL
# ==========================================

resnet_model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="COVID19_ResNet50"
)


# ==========================================
# 6. COMPILE MODEL
# ==========================================

resnet_model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 7. DISPLAY MODEL SUMMARY
# ==========================================

print("\n================================")
print("RESNET50 MODEL SUMMARY")
print("================================")

resnet_model.summary()

print("\nResNet50 architecture created successfully!")
