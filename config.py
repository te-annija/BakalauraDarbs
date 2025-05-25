import os
import tensorflow as tf

# Configuration constants

# Model training parameters
EFFICIENTNET_VERSION = "B3"
IMAGE_SIZE = 224
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
EPOCHS = 100

# Dataset directories
DATA_DIR = "dataset"

MODE_TRAIN = "train"
MODE_VAL = "val"
MODE_TEST = "test"
MODES = [MODE_TRAIN, MODE_VAL, MODE_TEST]

IMG_DIR_RAW = os.path.join(DATA_DIR, "pascal", "Images")
ANN_DIR_RAW = os.path.join(DATA_DIR, "pascal", "Annotations")
RAW_DATA_FOLDERS = ["car_pascal", "car_imagenet"]

# Utities
RANDOM_SEED = 42
BIN_COUNT = 8
BIN_WIDTH = 360 / BIN_COUNT

# Task parameters
LOCALIZATION_TASK = "localization"
ANGLE_CLASSIFICATION_TASK = "angle_classification"
ANGLE_REGRESSION_TASK = "angle_regression"
MTL_CLASSIFICATION = "mtl_classification"
MTL_REGRESSION = "mtl_regression"

TASKS = [
    LOCALIZATION_TASK,
    ANGLE_CLASSIFICATION_TASK,
    ANGLE_REGRESSION_TASK,
    MTL_CLASSIFICATION,
    MTL_REGRESSION,
]

TASK_CONFIGS_MAPPING = {
    LOCALIZATION_TASK: {
        "loss": tf.keras.losses.MeanSquaredError(),
        "metrics": [tf.keras.metrics.MeanSquaredError()],
    },
    ANGLE_CLASSIFICATION_TASK: {
        "loss": tf.keras.losses.SparseCategoricalCrossentropy(),
        "metrics": [tf.keras.metrics.SparseCategoricalAccuracy()],
    },
    ANGLE_REGRESSION_TASK: {
        "loss": tf.keras.losses.Huber(),
        "metrics": [tf.keras.metrics.MeanSquaredError()],
    },
    MTL_CLASSIFICATION: {
        "loss": [
            tf.keras.losses.MeanSquaredError(),
            tf.keras.losses.SparseCategoricalCrossentropy(),
        ],
        "loss_uw": [
            tf.keras.losses.MeanSquaredError(),
            tf.keras.losses.CategoricalCrossentropy(),
        ],
        "metrics": [
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ],
        "tasks": [LOCALIZATION_TASK, ANGLE_CLASSIFICATION_TASK],
    },
    MTL_REGRESSION: {
        "loss": [tf.keras.losses.MeanSquaredError(), tf.keras.losses.Huber()],
        "loss_uw": [tf.keras.losses.MeanSquaredError(), tf.keras.losses.Huber()],
        "metrics": [
            tf.keras.metrics.MeanSquaredError(),
            tf.keras.metrics.MeanSquaredError(),
        ],
        "tasks": [LOCALIZATION_TASK, ANGLE_REGRESSION_TASK],
    },
}

# MTL optimization parameters
BALANCER_UW = "uw"
BALANCER_GRADNORM = "gradnorm"
BALANCER_DWA = "dwa"
BALANCER_PCGRAD = "pcgrad"

BALANCERS = [BALANCER_UW, BALANCER_GRADNORM, BALANCER_DWA, BALANCER_PCGRAD]
