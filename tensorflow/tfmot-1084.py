import tensorflow as tf
import numpy as np
from Adder_Layer import Add2D
import tensorflow_model_optimization as tfmot

class GradientClippingModel(tf.keras.Model):
    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        metrics=None,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None,
        jit_compile=None,
        **kwargs,
    ):
        super().compile(optimizer, loss, metrics, loss_weights, weighted_metrics, run_eagerly, steps_per_execution, jit_compile, **kwargs)
        self.multiplier = {}
        for var in self.trainable_variables:
            if "conv2d" in var.name:
              self.multiplier[ var.name ] = 0.1*np.sqrt(np.product(var.shape))
            elif "dense" in var.name:
              self.multiplier[ var.name ] = 0.1*np.sqrt(np.product(var.shape))
            elif "batch_normalization" in var.name:
              self.multiplier[ var.name ] = 1.0
            elif "quantize_annotate" in var.name:
              self.multiplier[ var.name ] = 0.1*np.sqrt(np.product(var.shape))
            else:
              raise ValueError("layer name can't recognize, found {}".format(var.name))

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # adaptive scaling
        gradients = [grad / tf.norm(grad, ord=2) * self.multiplier[ trainable_vars[idx].name ] for idx,grad in enumerate(gradients)]
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    