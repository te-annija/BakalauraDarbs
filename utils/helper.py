from config import EFFICIENTNET_VERSION
import os
import tensorflow as tf


def get_model_path(task, get_weights=False):
    if get_weights:
        return f"results/models/{task}_efficientnet_{EFFICIENTNET_VERSION.lower()}_heads.weights.h5"
    else:
        return f"results/models/{task}_efficientnet_{EFFICIENTNET_VERSION.lower()}.keras"

def load_model(task):
    model_path = get_model_path(task)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    return model

