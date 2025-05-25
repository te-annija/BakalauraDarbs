import tensorflow as tf
from tensorflow import keras

class DWAUpdateCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if not self.model.use_dwa:
            return tf.constant([1.0 for _ in self.model.tasks], dtype=tf.float32)

        if any(len(self.model.loss_history[task]) < 2 for task in self.model.tasks):
            return tf.constant([1.0 for _ in self.model.tasks], dtype=tf.float32)

        r = []
        for task in self.model.tasks:
            prev_loss = self.model.loss_history[task][-1]
            prev_prev_loss = self.model.loss_history[task][-2]
            r_val = prev_loss / (prev_prev_loss + 1e-8)
            r.append(r_val)

        r = tf.constant(r, dtype=tf.float32)
        exp_r = tf.exp(r / 2.0)
        weights = exp_r / tf.reduce_sum(exp_r)
        self.model.current_weights.assign(weights)
        return weights

    def on_epoch_end(self, epoch, logs=None):
        if self.model.use_dwa:
            for i, task in enumerate(self.model.tasks):
                loss_val = self.model.task_loss_tracker[i].result().numpy()
                self.model.loss_history[task].append(loss_val)
                if len(self.model.loss_history[task]) > 2:
                    self.model.loss_history[task].pop(0)
