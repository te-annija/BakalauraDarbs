import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from dataset import Pascal3DDataset, adapt_for_uncertainty_mtl
from config import *
from model import CustomEfficientNet
from utils.helper import get_model_path
from utils.mtl_optimizers.dwa import DWAUpdateCallback
from test import test_model

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)


def get_callbacks(patience=3, mtl_weight_balancer=None):
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    if mtl_weight_balancer == BALANCER_DWA:
        callbacks.append(DWAUpdateCallback())

    return callbacks


def train_model(model_wrapper, train_ds, val_ds, epochs=30):
    model = model_wrapper.model
    model_phase_one_weights_path = get_model_path(model_wrapper.task, get_weights=True)

    # Load phase one wegihts if saved, train otherwise
    if os.path.exists(model_phase_one_weights_path):
        model.load_weights(model_phase_one_weights_path)
    else:
        model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=get_callbacks(patience=6),
            initial_epoch=0,
            verbose=2,
        )
        model.save_weights(get_model_path(model_wrapper.task, get_weights=True))

    #Phase two training
    if model_wrapper.mtl_weight_balancer == BALANCER_UW:
        train_ds = adapt_for_uncertainty_mtl(train_ds, model_wrapper.task)
        val_ds = adapt_for_uncertainty_mtl(val_ds, model_wrapper.task)
        model = model_wrapper.uw_trainable_model

    model = model_wrapper.unfreeze_model()
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=get_callbacks(
            patience=10, mtl_weight_balancer=model_wrapper.mtl_weight_balancer
        ),
        verbose=2,
    )

def main(task, weights, mtl_weight_balancer=None):
    dataset = Pascal3DDataset(mode="train", shuffle=True, augment=True)
    val_dataset = Pascal3DDataset(mode="val", shuffle=False)
    test_dataset = Pascal3DDataset(mode="test", shuffle=False)

    train_ds = dataset.get_dataset(task=task)
    val_ds = val_dataset.get_dataset(task=task)

    model_wrapper = CustomEfficientNet(task=task, weights=weights, mtl_weight_balancer=mtl_weight_balancer)

    train_model(model_wrapper=model_wrapper, train_ds=train_ds, val_ds=val_ds, epochs=EPOCHS)
    test_model(model_wrapper.model, task, test_dataset)

    if not weights and not mtl_weight_balancer:
        model_wrapper.model.save(get_model_path(task))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train EfficientNet model for Pascal3D+ tasks."
    )
    parser.add_argument("--task", type=str, choices=TASKS, default=MTL_CLASSIFICATION)
    parser.add_argument("--weights", type=float, nargs="+", default=None)
    parser.add_argument("--balancer", type=str, choices=BALANCERS, default=None)

    args = parser.parse_args()

    main(task=args.task, weights=args.weights, mtl_weight_balancer=args.balancer)
