import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetV2B3 as EfficientNet

from config import *
from utils.custom_mtl_model import CustomMTLModel
from utils.mtl_optimizers.uncertanty_weighing import CustomMultiLossLayer
from utils.mtl_optimizers.gradnorm import GradNormModel

tf.random.set_seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)

class CustomEfficientNet:
    def __init__(self, task, weights=None, mtl_weight_balancer=None):
        self.task = task
        self.mtl_weight_balancer = mtl_weight_balancer
        self.loss_weights = weights

        config = TASK_CONFIGS_MAPPING[self.task]
        self.loss = config["loss"]
        self.metrics = config["metrics"]
        self.tasks = config["tasks"] if "tasks" in config else self.task

        self.base_model = self._prepare_base_model()
        self.heads = self._prepare_heads()
        self.model = self._build_model()

        self.uw_trainable_model = (
            self._build_uw_trainable_model()
            if mtl_weight_balancer == BALANCER_UW
            else None
        )

    def _prepare_base_model(self):
        base_model = EfficientNet(
            weights="imagenet",
            include_top=False,
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        )
        base_model.trainable = False
        return base_model

    def _prepare_heads(self):
        if self.task == MTL_CLASSIFICATION:
            return [
                self._prepare_localization_head(),
                self._prepare_classification_head(),
            ]
        elif self.task == MTL_REGRESSION:
            return [
                self._prepare_localization_head(),
                self._prepare_angle_regression_head(),
            ]
        elif self.task == LOCALIZATION_TASK:
            return self._prepare_localization_head()
        elif self.task == ANGLE_CLASSIFICATION_TASK:
            return self._prepare_classification_head()
        elif self.task == ANGLE_REGRESSION_TASK:
            return self._prepare_angle_regression_head()

    def _prepare_localization_head(self):
        x = GlobalAveragePooling2D()(self.base_model.output)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        x = Dense(4, name=LOCALIZATION_TASK, activation="sigmoid")(x)
        return x

    def _prepare_classification_head(self):
        x = GlobalAveragePooling2D()(self.base_model.output)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(BIN_COUNT, name=ANGLE_CLASSIFICATION_TASK, activation="softmax")(x)
        return x

    def _prepare_angle_regression_head(self):
        x = GlobalAveragePooling2D()(self.base_model.output)
        x = Dense(1024, activation="relu")(x)
        x = Dropout(0.3)(x)
        x = Dense(2, name=ANGLE_REGRESSION_TASK, activation="linear")(x)
        return x

    def _compile_model(self, model, learning_rate, **kwargs):
        compile_args = {
            "optimizer": tf.keras.optimizers.Adam(learning_rate=learning_rate),
            "loss": self.loss,
            "metrics": self.metrics,
            "loss_weights": self.loss_weights,
        }
        compile_args.update(kwargs)
        model.compile(**compile_args)
        return model

    def _build_model(self):
        if self.mtl_weight_balancer == BALANCER_GRADNORM:
            model = GradNormModel(inputs=self.base_model.input, outputs=self.heads, tasks=self.tasks)
        elif self.mtl_weight_balancer in [BALANCER_PCGRAD, BALANCER_DWA]:
            model = CustomMTLModel(inputs=self.base_model.input, outputs=self.heads, tasks=self.tasks)
        else:
            model = tf.keras.Model(inputs=self.base_model.input, outputs=self.heads)

        return self._compile_model(model, learning_rate=LEARNING_RATE, loss_weights=self.loss_weights)

    # Uncertainty Weighing adaptation from yaringal/multi-task-learning-example
    def _build_uw_trainable_model(self):
        prediction_model = self.model
        image_input = tf.keras.Input(
            shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image_input"
        )

        y_true1 = tf.keras.Input(shape=(4,), name="y_true1")
        y_true2_shape = (BIN_COUNT,) if self.task == MTL_CLASSIFICATION else (2,)
        y_true2 = tf.keras.Input(shape=y_true2_shape, name="y_true2")

        y_pred1, y_pred2 = prediction_model(image_input)
        output = CustomMultiLossLayer(
            loss_fns=TASK_CONFIGS_MAPPING[self.task]["loss_uw"], nb_outputs=2
        )([y_true1, y_true2, y_pred1, y_pred2])

        model = tf.keras.Model(inputs=[image_input, y_true1, y_true2], outputs=output)
        return model

    def unfreeze_model(self):
        self.base_model.trainable = True

        for layer in self.base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        learning_rate = LEARNING_RATE / 5

        if self.mtl_weight_balancer == BALANCER_UW:
            self.uw_trainable_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=None,
            )
            return self.uw_trainable_model
        elif self.mtl_weight_balancer == BALANCER_GRADNORM:
            return self._compile_model(self.model, learning_rate, use_grad_norm=True)
        elif self.mtl_weight_balancer == BALANCER_DWA:
            return self._compile_model(self.model, learning_rate, use_dwa=True)
        elif self.mtl_weight_balancer == BALANCER_PCGRAD:
            return self._compile_model(self.model, learning_rate, use_pcgrad=True)

        return self._compile_model(self.model, learning_rate)

if __name__ == "__main__":
    model = CustomEfficientNet(task=MTL_REGRESSION)
    model.model.summary()
