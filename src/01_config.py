
# ==========================================
# PROJECT CONFIGURATION
# ==========================================

# Image configuration
IMAGE_SIZE = (224, 224)
CHANNELS = 1

# Dataset configuration
CLASS_NAMES = [
    "COVID",
    "Normal",
    "Viral Pneumonia",
    "Lung_Opacity"
]

NUM_CLASSES = len(CLASS_NAMES)

# Training configuration
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.0001

# Reproducibility
RANDOM_SEED = 42
