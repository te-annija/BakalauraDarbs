# PCGrad implementation adapted and modified from tianheyu927/PCGrad
# Original: https://github.com/tianheyu927/PCGrad (MIT Licensed)
# Copyright (c) 2020 Tianhe Yu

import tensorflow as tf

class PCGradOptimizer:
    def __init__(self):
        pass
    def compute_gradients(self, losses, var_list):
        num_tasks = len(losses)
        
        random_indices = tf.random.shuffle(tf.range(num_tasks))
        losses = tf.gather(losses, random_indices)
        
        stacked_losses = tf.stack(losses)
        
        grads_task_list = []
        for i in range(num_tasks):
            task_loss = stacked_losses[i]
            grad_list = tf.gradients(task_loss, var_list)
            flat_grads = []
            for g in grad_list:
                if g is not None:
                    flat_grads.append(tf.reshape(g, [-1]))
                else:
                    var_shape = var_list[len(flat_grads)].shape
                    flat_grads.append(tf.zeros(tf.reduce_prod(var_shape), dtype=task_loss.dtype))
            
            grads_task_list.append(tf.concat(flat_grads, axis=0))
        
        grads_task = tf.stack(grads_task_list)
        
        def project_gradients(grad_task):
            proj_grad = grad_task
            for k in range(num_tasks):
                inner_product = tf.reduce_sum(proj_grad * grads_task[k])
                denominator = tf.reduce_sum(grads_task[k] * grads_task[k])
                safe_denominator = tf.maximum(denominator, 1e-10)
                proj_direction = inner_product / safe_denominator
                proj_grad = proj_grad - tf.minimum(proj_direction, 0.) * grads_task[k]
            return proj_grad
        
        projected_grads_flat = []
        for j in range(num_tasks):
            projected_grads_flat.append(project_gradients(grads_task[j]))
        
        summed_proj_grads_flat = tf.zeros_like(grads_task[0])
        for j in range(num_tasks):
            summed_proj_grads_flat += projected_grads_flat[j]
        
        start_idx = 0
        final_grads = []
        for var in var_list:
            var_shape = var.shape
            flatten_dim = tf.reduce_prod(var_shape)
            var_flat_size = tf.cast(flatten_dim, tf.int32)
            
            proj_grad = summed_proj_grads_flat[start_idx:start_idx + var_flat_size]
            proj_grad = tf.reshape(proj_grad, var_shape)
            final_grads.append(proj_grad)
            
            start_idx += var_flat_size
        
        grads_and_vars = list(zip(final_grads, var_list))
        
        return grads_and_vars