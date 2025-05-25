
import tensorflow as tf
from tensorflow import keras
from mtl_optimizers.pcgrad import PCGradOptimizer

class CustomMTLModel(keras.Model):
    def __init__(self, *args, **kwargs):
        self.tasks = kwargs.pop("tasks", False)
        super().__init__(*args, **kwargs)

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.task_loss_tracker = [keras.metrics.Mean(name=task) for task in self.tasks]
        self.loss_history = {task: [] for task in self.tasks}
        self.current_weights = tf.Variable([1.0 for _ in self.tasks], dtype=tf.float32)
    
    @property
    def metrics(self):
        return [self.loss_tracker] + self.t_metrics + self.task_loss_tracker
    
    def compile(self, *args, **kwargs):
        self.use_dwa = kwargs.pop("use_dwa", False)
        self.use_pcgrad = kwargs.pop("use_pcgrad", False)
        self.pcgrad = PCGradOptimizer() if self.use_pcgrad else None

        super().compile(*args, **kwargs)

        self.t_loss_fn = kwargs.get("loss", None)
        self.t_metrics = kwargs.get("metrics", None)

    def compute_loss(self, y_true, y_pred):
        task_losses = []
        for i, task in enumerate(self.tasks):
            loss = self.t_loss_fn[i](y_true[task], y_pred[i])
            loss = tf.reduce_mean(loss)
            task_losses.append(loss)
                
        weighted_losses = [self.current_weights[i] * task_losses[i] for i in range(len(task_losses))]
        total_loss = tf.reduce_sum(tf.stack(weighted_losses))

        return total_loss, task_losses

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)
            loss, task_losses = self.compute_loss(y, y_pred)

        trainable_vars = self.trainable_variables
        if self.use_pcgrad:
            gradients_and_vars = self.pcgrad.compute_gradients(task_losses, trainable_vars)
            self.optimizer.apply_gradients(gradients_and_vars)
        else:
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        del tape

        self.loss_tracker.update_state(loss)
        for i, task in enumerate(self.tasks):
            self.task_loss_tracker[i].update_state(task_losses[i])
            self.t_metrics[i].update_state(y[task], y_pred[i])

        metrics = {m.name: m.result() for m in self.metrics}
        metrics["w1"] = self.current_weights[0]
        metrics["w2"] = self.current_weights[1]
   
        return metrics
    
    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)

        loss, task_losses = self.compute_loss(y, y_pred)

        self.loss_tracker.update_state(loss)
        for i, task in enumerate(self.tasks):
            self.t_metrics[i].update_state(y[task], y_pred[i])
            self.task_loss_tracker[i].update_state(task_losses[i])
        
        return {m.name: m.result() for m in self.metrics}
