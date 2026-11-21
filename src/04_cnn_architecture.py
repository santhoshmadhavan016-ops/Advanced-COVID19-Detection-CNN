
import tensorflow as tf

from tensorflow.keras import layers, models

# ==========================================
# MODEL CONFIGURATION
# ==========================================

IMAGE_SIZE = (224, 224)
CHANNELS = 1
NUM_CLASSES = 4


# ==========================================
# CUSTOM CNN ARCHITECTURE
# ==========================================

def build_custom_cnn():

    # Input layer
    inputs = layers.Input(
        shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)
    )

    # Convolution Block 1
    x = layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    )(inputs)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolution Block 2
    x = layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Convolution Block 3
    x = layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Feature extraction
    x = layers.GlobalAveragePooling2D()(x)

    # Classification layers
    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.5)(x)

    # Output layer
    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    # Create model
    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Advanced_Custom_CNN"
    )

    return model


# ==========================================
# MODEL SUMMARY
# ==========================================

if __name__ == "__main__":

    model = build_custom_cnn()

    model.summary()
