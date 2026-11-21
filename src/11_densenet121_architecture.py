
import tensorflow as tf

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (
    Input,
    Concatenate,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Rescaling,
    Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import preprocess_input


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

IMAGE_SIZE = 224
NUM_CLASSES = 4


# --------------------------------------------------
# 2. BUILD DENSENET121
# --------------------------------------------------

def build_densenet121():

    # Grayscale input
    inputs = Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
        name="grayscale_input"
    )

    # Convert grayscale image into 3 channels
    x = Concatenate(axis=-1, name="rgb_conversion")(
        [inputs, inputs, inputs]
    )

    # Convert pixel values from [0, 1] to [0, 255]
    x = Rescaling(
        scale=255.0,
        name="pixel_rescaling"
    )(x)

    # DenseNet preprocessing
    x = Lambda(
        preprocess_input,
        name="densenet_preprocessing"
    )(x)

    # Load pretrained DenseNet121
    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    x = base_model(x, training=False)

    # Classification head
    x = GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = Dense(
        128,
        activation="relu",
        name="dense_layer"
    )(x)

    x = Dropout(
        0.5,
        name="dropout"
    )(x)

    outputs = Dense(
        NUM_CLASSES,
        activation="softmax",
        name="predictions"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="DenseNet121_COVID_Classifier"
    )

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# --------------------------------------------------
# 3. BUILD MODEL
# --------------------------------------------------

if __name__ == "__main__":

    model = build_densenet121()

    print("\nDenseNet121 Architecture")
    print("========================")

    model.summary()

    print("\nTotal parameters:", model.count_params())

    print("\nDenseNet121 architecture created successfully!")
