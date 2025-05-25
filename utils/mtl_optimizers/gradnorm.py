# GradNorm implementation adapted and modified from jpcastillog/GradNorm-Keras
# Original: https://github.com/jpcastillog/GradNorm-Keras

import tensorflow as tf
from tensorflow import keras

class GradNormModel(keras.Model):
    def __init__(self, *args, **kwargs):
        self.tasks = kwargs.pop("tasks", None)
        
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_w_tracker = [keras.metrics.Mean(name=task) for task in self.tasks]
        self.w1 = tf.Variable(1.0, trainable=True, dtype=tf.float32,  constraint=tf.keras.constraints.NonNeg())
        self.w2 = tf.Variable(1.0, trainable=True, dtype=tf.float32,  constraint=tf.keras.constraints.NonNeg())
        
        self.l01 = tf.Variable(-1.0, trainable=False, dtype=tf.float32)
        self.l02 = tf.Variable(-1.0, trainable=False, dtype=tf.float32)

        self.alpha = 0.25
        self.gradnorm_optimizer = keras.optimizers.Adam(learning_rate=0.001)

    def compile(self, *args, **kwargs):
        self.use_grad_norm = kwargs.pop("use_grad_norm", False)
        super().compile(*args, **kwargs)
        self.t_loss_fn = kwargs.get("loss", None)
        self.t_metrics = kwargs.get("metrics", None)
    
    def compute_loss(self, y_true, y_pred):
        loss_t1 = self.t_loss_fn[0](y_true[self.tasks[0]], y_pred[0])
        loss_t2 = self.t_loss_fn[1](y_true[self.tasks[1]], y_pred[1])

        l1 = tf.multiply(loss_t1, self.w1)
        l2 = tf.multiply(loss_t2, self.w2)

        loss = tf.add(l1, l2)

        if self.use_grad_norm:
            self.l01.assign(tf.cond(self.l01 < 0, lambda: loss_t1, lambda: self.l01))
            self.l02.assign(tf.cond(self.l02 < 0, lambda: loss_t2, lambda: self.l02))

        return loss, l1, l2
    
    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss, l1, l2 = self.compute_loss(y, y_pred)
            
            if self.use_grad_norm:
                last_shared_layer = self.get_layer('top_conv')
                weights_shared = last_shared_layer.trainable_weights

                G1R = tape.gradient(l1, weights_shared)
                G1 = tf.linalg.global_norm(G1R)

                G2R = tape.gradient(l2, weights_shared)
                G2 = tf.linalg.global_norm(G2R)
                G_avg = tf.divide(tf.add(G1, G2), 2)

                l_hat_1 = tf.divide(l1, self.l01)
                l_hat_2 = tf.divide(l2, self.l02)
                l_hat_avg = tf.divide(tf.add(l_hat_1, l_hat_2), 2)

                inv_rate_1 = tf.divide(l_hat_1, l_hat_avg)
                inv_rate_2 = tf.divide(l_hat_2, l_hat_avg)

                a = tf.constant(self.alpha)
                C1 = tf.multiply(G_avg, tf.pow(inv_rate_1, a))
                C2 = tf.multiply(G_avg, tf.pow(inv_rate_2, a))
                C1 = tf.stop_gradient(tf.identity(C1))
                C2 = tf.stop_gradient(tf.identity(C2))

                loss_gradnorm = tf.add(
                    tf.reduce_sum(tf.abs(tf.subtract(G1, C1))),
                    tf.reduce_sum(tf.abs(tf.subtract(G2, C2))))

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        if self.use_grad_norm:
            tape.watch([self.w1, self.w2])
            gradnorm_grads = tape.gradient(loss_gradnorm, [self.w1, self.w2])
            gradnorm_grads = [tf.clip_by_value(g, -1.0, 1.0) for g in gradnorm_grads]
            self.gradnorm_optimizer.apply_gradients(zip(gradnorm_grads, [self.w1, self.w2]))
            epsilon = 1e-4
            self.w1.assign(tf.maximum(self.w1, epsilon))
            self.w2.assign(tf.maximum(self.w2, epsilon))

            coef = tf.divide(2.0, tf.add(self.w1, self.w2))
            self.w1.assign(tf.multiply(self.w1, coef))
            self.w2.assign(tf.multiply(self.w2, coef))

        del tape  

        self.loss_tracker.update_state(loss)
        self.loss_w_tracker[0].update_state(l1)
        self.loss_w_tracker[1].update_state(l2)

        for i, task in enumerate(self.tasks):
            self.t_metrics[i].update_state(y[task], y_pred[i])

        metrics = {m.name: m.result() for m in self.metrics}
        metrics["w1"] = self.w1.read_value()
        metrics["w2"] = self.w2.read_value()
        return metrics
    @property
    def metrics(self):
        return [self.loss_tracker] + self.loss_w_tracker + self.t_metrics
    def test_step(self, data):
            x, y = data
            y_pred = self(x, training=False)

            loss, l1, l2 = self.compute_loss(y, y_pred)

            self.loss_tracker.update_state(loss)
            self.loss_w_tracker[0].update_state(l1)
            self.loss_w_tracker[1].update_state(l2)
            
            for i, task in enumerate(self.tasks):
                self.t_metrics[i].update_state(y[task], y_pred[i])
            
            return {m.name: m.result() for m in self.metrics}