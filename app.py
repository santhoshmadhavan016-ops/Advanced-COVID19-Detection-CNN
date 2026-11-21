
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import importlib
import sys

from pathlib import Path
from PIL import Image
from huggingface_hub import hf_hub_download


# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Chest X-ray Classification",
    page_icon="🩻",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
SRC_DIR = BASE_DIR / "src"

HF_REPO_ID = "santhoshmadhavan016/covid19-detection-models"

CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]

MODEL_NAMES = [
    "Custom CNN",
    "ResNet50",
    "DenseNet121"
]

MODEL_METRICS = {
    "Custom CNN": {
        "Accuracy": 91.81,
        "Precision": 0.9254,
        "Recall": 0.9315,
        "F1-score": 0.9283,
        "ROC-AUC": 0.9875
    },
    "ResNet50": {
        "Accuracy": 91.81,
        "Precision": 0.9258,
        "Recall": 0.9213,
        "F1-score": 0.9235,
        "ROC-AUC": 0.9871
    },
    "DenseNet121": {
        "Accuracy": 87.09,
        "Precision": 0.8658,
        "Recall": 0.8907,
        "F1-score": 0.8774,
        "ROC-AUC": 0.9746
    }
}


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: 750;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #999999;
    font-size: 16px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 10px;
}

.result-card {
    padding: 22px;
    border-radius: 12px;
    background-color: rgba(128,128,128,0.10);
    text-align: center;
    border: 1px solid rgba(128,128,128,0.25);
}

.result-label {
    font-size: 15px;
    color: #888888;
}

.result-value {
    font-size: 27px;
    font-weight: bold;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. HEADER
# ==================================================

st.markdown(
    '<div class="main-title">🩻 AI Chest X-ray Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Deep Learning Based COVID-19 and Lung Condition Classification</div>',
    unsafe_allow_html=True
)

st.divider()


# ==================================================
# 4. MODEL FILE DOWNLOAD
# ==================================================

@st.cache_data
def download_model_file(filename):

    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename
    )


# ==================================================
# 5. MODEL LOADING
# ==================================================

@st.cache_resource
def load_model(model_name):

    if model_name == "Custom CNN":

        model_path = download_model_file(
            "custom_cnn_best.keras"
        )

        return tf.keras.models.load_model(
            model_path,
            compile=False
        )

    if model_name == "ResNet50":

        model_path = download_model_file(
            "resnet50_best.keras"
        )

        preprocess_input = (
            tf.keras.applications.resnet50.preprocess_input
        )

        return tf.keras.models.load_model(
            model_path,
            custom_objects={
                "preprocess_input": preprocess_input
            },
            compile=False
        )

    if model_name == "DenseNet121":

        weights_path = download_model_file(
            "densenet121_best_v2.weights.h5"
        )

        sys.path.insert(0, str(SRC_DIR))

        module = importlib.import_module(
            "11_densenet121_architecture"
        )

        model = module.build_densenet121()

        model.load_weights(weights_path)

        return model

    raise ValueError("Unknown model selected")


# ==================================================
# 6. IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image):

    image = image.convert("L")

    image = image.resize((224, 224))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ==================================================
# 7. PREDICTION
# ==================================================

def predict_image(model, image_array):

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[predicted_index] * 100
    )

    return predicted_class, confidence, predictions


# ==================================================
# 8. GRAD-CAM
# ==================================================

def generate_gradcam(model, model_name, image_array):

    input_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    if model_name == "Custom CNN":

        last_conv = model.get_layer("conv2d_2")

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[
                last_conv.output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(
                input_tensor,
                training=False
            )

            class_index = tf.argmax(
                predictions[0]
            )

            class_score = predictions[:, class_index]

        gradients = tape.gradient(
            class_score,
            conv_outputs
        )

    else:

        if model_name == "ResNet50":

            backbone_name = "resnet50"

            preprocess_input = (
                tf.keras.applications.resnet50.preprocess_input
            )

        else:

            backbone_name = "densenet121"

            preprocess_input = (
                tf.keras.applications.densenet.preprocess_input
            )

        backbone = model.get_layer(backbone_name)

        last_conv = None

        for layer in reversed(backbone.layers):

            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            ):

                last_conv = layer
                break

        if last_conv is None:
            raise ValueError("Convolutional layer not found")

        backbone_grad_model = tf.keras.Model(
            inputs=backbone.input,
            outputs=[
                last_conv.output,
                backbone.output
            ]
        )

        backbone_index = model.layers.index(backbone)

        head_layers = model.layers[backbone_index + 1:]

        rgb_input = tf.repeat(
            input_tensor,
            repeats=3,
            axis=-1
        )

        rgb_input = rgb_input * 255.0

        processed_input = preprocess_input(rgb_input)

        with tf.GradientTape() as tape:

            conv_outputs, features = backbone_grad_model(
                processed_input,
                training=False
            )

            predictions = features

            for layer in head_layers:

                if isinstance(
                    layer,
                    tf.keras.layers.Dropout
                ):

                    predictions = layer(
                        predictions,
                        training=False
                    )

                else:

                    predictions = layer(predictions)

            class_index = tf.argmax(
                predictions[0]
            )

            class_score = predictions[:, class_index]

        gradients = tape.gradient(
            class_score,
            conv_outputs
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        (224, 224)
    ).numpy().squeeze()

    return heatmap


# ==================================================
# 9. DISPLAY GRAD-CAM
# ==================================================

def display_gradcam(original_image, heatmap):

    original = np.array(
        original_image.convert("RGB").resize((224, 224))
    )

    color_map = plt.get_cmap("jet")

    heatmap_color = color_map(heatmap)[:, :, :3]

    heatmap_color = (
        heatmap_color * 255
    ).astype(np.uint8)

    overlay = (
        0.55 * original.astype(np.float32)
        + 0.45 * heatmap_color.astype(np.float32)
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            original,
            caption="Original X-ray",
            use_container_width=True
        )

    with col2:
        st.image(
            heatmap_color,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )

    with col3:
        st.image(
            overlay,
            caption="Grad-CAM Overlay",
            use_container_width=True
        )


# ==================================================
# 10. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("🩻 AI X-ray System")

st.sidebar.caption(
    "Deep Learning Classification Platform"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "X-ray Prediction",
        "Model Evaluation",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Academic Research Project\n\n"
    "COVID-19 and Lung Condition Classification"
)


# ==================================================
# 11. DASHBOARD
# ==================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Project Dashboard</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Overview of the dataset, deep learning models "
        "and experimental evaluation."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dataset Images", "21,165")

    col2.metric("Deep Learning Models", "3")

    col3.metric("Classification Classes", "4")

    col4.metric("Test Images", "3,175")

    st.divider()

    st.subheader("Classification Categories")

    class_cols = st.columns(4)

    for index, class_name in enumerate(CLASS_NAMES):

        class_cols[index].markdown(
            f"**{class_name}**"
        )

    st.divider()

    st.subheader("Model Performance Overview")

    metric_cols = st.columns(3)

    for index, model_name in enumerate(MODEL_NAMES):

        metric_cols[index].metric(
            model_name,
            f"{MODEL_METRICS[model_name]['Accuracy']:.2f}%",
            "Test accuracy"
        )

    st.subheader("Accuracy Comparison")

    accuracy_path = (
        RESULTS_DIR
        / "comparison"
        / "model_accuracy_comparison.png"
    )

    if accuracy_path.exists():

        st.image(
            str(accuracy_path),
            use_container_width=True
        )

    else:

        st.warning("Accuracy comparison graph not found.")

    st.subheader("Combined ROC Curve")

    roc_path = (
        RESULTS_DIR
        / "comparison"
        / "combined_three_model_roc_curve.png"
    )

    if roc_path.exists():

        st.image(
            str(roc_path),
            use_container_width=True
        )

    else:

        st.warning("Combined ROC curve not found.")


# ==================================================
# 12. X-RAY PREDICTION
# ==================================================

elif page == "X-ray Prediction":

    st.markdown(
        '<div class="section-title">X-ray Prediction</div>',
        unsafe_allow_html=True
    )

    selected_model = st.selectbox(
        "Select Deep Learning Model",
        MODEL_NAMES + ["Compare All Models"]
    )

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray Image",
        type=["png", "jpg", "jpeg"],
        key="dashboard_xray_uploader"
    )

    if uploaded_file is not None:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.image(
            original_image,
            caption="Uploaded Chest X-ray",
            width=350
        )

        image_array = preprocess_image(
            original_image
        )

        st.divider()

        if selected_model != "Compare All Models":

            st.subheader(
                f"{selected_model} Prediction"
            )

            try:

                with st.spinner(
                    f"Analyzing with {selected_model}..."
                ):

                    model = load_model(selected_model)

                    predicted_class, confidence, predictions = (
                        predict_image(model, image_array)
                    )

                st.success("Analysis completed")

                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">Predicted Condition</div>
                            <div class="result-value">{predicted_class}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with result_col2:

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">Prediction Confidence</div>
                            <div class="result-value">{confidence:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.subheader("Class Probability Scores")

                for index, class_name in enumerate(CLASS_NAMES):

                    score = float(
                        predictions[index] * 100
                    )

                    st.write(
                        f"{class_name}: {score:.2f}%"
                    )

                    st.progress(score / 100)

                st.divider()

                st.subheader("Grad-CAM Visualization")

                with st.spinner("Generating Grad-CAM..."):

                    heatmap = generate_gradcam(
                        model,
                        selected_model,
                        image_array
                    )

                display_gradcam(
                    original_image,
                    heatmap
                )

            except Exception as error:

                st.error(
                    f"Prediction failed: {error}"
                )

        else:

            st.subheader("Three-Model Prediction Comparison")

            comparison_results = []

            progress = st.progress(0)

            try:

                for index, model_name in enumerate(MODEL_NAMES):

                    with st.spinner(
                        f"Running {model_name}..."
                    ):

                        model = load_model(model_name)

                        predicted_class, confidence, predictions = (
                            predict_image(model, image_array)
                        )

                        comparison_results.append({
                            "Model": model_name,
                            "Prediction": predicted_class,
                            "Confidence": f"{confidence:.2f}%"
                        })

                    progress.progress(
                        (index + 1) / 3
                    )

                st.success("All models completed successfully")

                st.table(comparison_results)

                st.subheader("Prediction Confidence Comparison")

                for result in comparison_results:

                    st.write(
                        f"{result['Model']} — "
                        f"{result['Prediction']} "
                        f"({result['Confidence']})"
                    )

                st.info(
                    "Select an individual model to view its Grad-CAM visualization."
                )

            except Exception as error:

                st.error(
                    f"Model comparison failed: {error}"
                )


# ==================================================
# 13. MODEL EVALUATION
# ==================================================

elif page == "Model Evaluation":

    st.markdown(
        '<div class="section-title">Model Evaluation</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Experimental evaluation of all three models "
        "using the same 3,175-image test dataset."
    )

    st.divider()

    st.subheader("Performance Metrics")

    metric_data = []

    for model_name in MODEL_NAMES:

        metrics = MODEL_METRICS[model_name]

        metric_data.append({
            "Model": model_name,
            "Accuracy (%)": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1-score": metrics["F1-score"],
            "ROC-AUC": metrics["ROC-AUC"]
        })

    st.dataframe(
        metric_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Classification Metrics Comparison")

    metrics_path = (
        RESULTS_DIR
        / "comparison"
        / "classification_metrics_comparison.png"
    )

    if metrics_path.exists():

        st.image(
            str(metrics_path),
            use_container_width=True
        )

    st.subheader("Accuracy Comparison")

    accuracy_path = (
        RESULTS_DIR
        / "comparison"
        / "model_accuracy_comparison.png"
    )

    if accuracy_path.exists():

        st.image(
            str(accuracy_path),
            use_container_width=True
        )

    st.subheader("ROC-AUC Comparison")

    auc_path = (
        RESULTS_DIR
        / "comparison"
        / "model_roc_auc_comparison.png"
    )

    if auc_path.exists():

        st.image(
            str(auc_path),
            use_container_width=True
        )

    st.subheader("Combined Three-Model ROC Curve")

    combined_roc_path = (
        RESULTS_DIR
        / "comparison"
        / "combined_three_model_roc_curve.png"
    )

    if combined_roc_path.exists():

        st.image(
            str(combined_roc_path),
            use_container_width=True
        )

    st.divider()

    st.subheader("Individual Model Evaluation")

    selected_evaluation_model = st.selectbox(
        "Choose model for detailed evaluation",
        MODEL_NAMES,
        key="evaluation_model"
    )

    folder_names = {
        "Custom CNN": "custom_cnn",
        "ResNet50": "resnet50",
        "DenseNet121": "densenet121"
    }

    model_folder = (
        RESULTS_DIR / folder_names[selected_evaluation_model]
    )

    confusion_path = model_folder / "confusion_matrix.png"

    roc_curve_path = model_folder / "roc_curve.png"

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("**Confusion Matrix**")

        if confusion_path.exists():

            st.image(
                str(confusion_path),
                use_container_width=True
            )

        else:

            st.warning("Confusion matrix not found.")

    with col2:

        st.markdown("**ROC Curve**")

        if roc_curve_path.exists():

            st.image(
                str(roc_curve_path),
                use_container_width=True
            )

        else:

            st.warning("ROC curve not found.")

    st.divider()

    st.info(
        "All results are experimental and based on the same test dataset. "
        "They do not establish clinical validation."
    )


# ==================================================
# 14. ABOUT PROJECT
# ==================================================

elif page == "About Project":

    st.markdown(
        '<div class="section-title">About the Project</div>',
        unsafe_allow_html=True
    )

    st.write("""
    This project presents a deep-learning-based system for
    classifying chest X-ray images into four lung-condition categories.
    """)

    st.subheader("Project Objectives")

    st.markdown("""
    - Develop a deep learning image classification system.
    - Compare three CNN-based architectures.
    - Evaluate performance using classification metrics.
    - Visualize important image regions using Grad-CAM.
    """)

    st.subheader("Dataset Information")

    st.write("Total dataset images: 21,165")

    dataset_data = {
        "Class": [
            "COVID",
            "Normal",
            "Viral Pneumonia",
            "Lung Opacity"
        ],
        "Images": [
            3616,
            10192,
            1345,
            6012
        ]
    }

    st.dataframe(
        dataset_data,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Deep Learning Models")

    st.markdown("""
    **1. Custom CNN**

    A custom convolutional neural network developed for
    four-class chest X-ray classification.

    **2. ResNet50**

    A residual neural network using transfer learning.

    **3. DenseNet121**

    A densely connected convolutional neural network
    using transfer learning.
    """)

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - TensorFlow and Keras
    - Streamlit
    - NumPy
    - Matplotlib
    - Scikit-learn
    - Pillow
    - Hugging Face Hub
    """)

    st.subheader("Explainable AI")

    st.write("""
    Grad-CAM is used to produce heatmaps that help visualize
    image regions influencing the model's predictions.
    """)

    st.divider()

    st.subheader("Medical Disclaimer")

    st.warning("""
    This application is an academic research prototype.

    Predictions are experimental and have not been clinically validated.
    This system is not a medical diagnostic tool and must not replace
    professional medical advice, diagnosis or treatment.
    """)


# ==================================================
# 15. FOOTER
# ==================================================

st.divider()

st.caption(
    "AI Chest X-ray Classification | Academic Research Project | "
    "Experimental results only — not for clinical diagnosis."
)
