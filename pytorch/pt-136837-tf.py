import tensorflow as tf
import numpy as np
from tqdm import tqdm

inp = tf.cast(tf.random.uniform((10, 32, 32, 1), maxval=2, dtype=tf.int32), tf.float32)
print(inp.shape)

class MaskedConv2DA(tf.keras.layers.Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size, padding="same", **kwargs):
        super().__init__(filters=out_channels, kernel_size=kernel_size, padding=padding, **kwargs)
        # self.kernel_size = kernel_size
        # self.mask = None

    def build(self, input_shape):
        super().build(input_shape)
        mask = np.zeros(self.kernel.shape, dtype=np.float32)
        # print("!!!",  self.kernel_size)
        center = self.kernel_size[0] // 2

        mask[:, :center, :, :] = 1
        mask[center, :center, :, :] = 1
        self.mask = tf.convert_to_tensor(mask, dtype=self.kernel.dtype)

    def call(self, x):
        # self.kernel.assign(self.kernel * self.mask)
        return super().call(x)

class MaskedConv2DB(MaskedConv2DA):
    def build(self, input_shape):
        super().build(input_shape)
        center = self.kernel_size[0] // 2
        self.mask = np.zeros(self.kernel.shape, dtype=np.float32)
        self.mask[:, :center, :, :] = 1
        self.mask[center, :center, :, :] = 1
        self.mask[center, center, :, :] = 1
        self.mask = tf.convert_to_tensor(self.mask, dtype=self.kernel.dtype)

class PixelCNN(tf.keras.Model):
    def __init__(self, num_channels, num_colors, H, W, n_layers=5):
        super(PixelCNN, self).__init__()
        self.num_channels = num_channels
        self.num_colors = num_colors
        self.H = H
        self.W = W

        kernel_size = 7
        padding = "same"  # Set padding to 'same' for consistency in TensorFlow

        # Define Masked Conv2D A Layer
        self.layers_list = [
            MaskedConv2DA(in_channels=num_channels, out_channels=64, kernel_size=kernel_size, padding=padding),
            tf.keras.layers.ReLU(),
        ]

        # Define Masked Conv2D B Layers
        for _ in range(n_layers):
            self.layers_list.extend([
                MaskedConv2DB(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=padding),
                tf.keras.layers.ReLU(),
            ])

        # Final Conv2D Layers
        self.layers_list.extend([
            tf.keras.layers.Conv2D(64, kernel_size=1, padding="valid"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(num_channels * num_colors, kernel_size=1, padding="valid"),
        ])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, [0, 3, 1, 2])  # (B, H, W, C) -> (B, C, H, W)
        x = 2.0 * (tf.cast(x, tf.float32) / self.num_colors) - 1.0  # Rescale to [-1, 1]

        print("------ Start")
        print(self.layers_list)

        for layer in self.layers_list:
            print(layer)
            x = layer(x)

        print("------ End")

        x = tf.transpose(x, [0, 2, 3, 1])  # (B, C, H, W) -> (B, H, W, C)
        return tf.reshape(x, (batch_size, self.H, self.W, self.num_channels, self.num_colors))  # Reshape to (B, H, W, C, K)

    def sample(self, num_samples):
        samples = tf.zeros((num_samples, self.H, self.W, self.num_channels), dtype=tf.float32)
        for i in tqdm(range(self.H), desc="Heights"):
            for j in tqdm(range(self.W), desc="Widths"):
                for k in range(self.num_channels):
                    logits = self.call(samples)[:, i, j, k, :]  # (B, H, W, C, K) -> (B, K)
                    prob = tf.nn.softmax(logits, axis=-1)
                    samples[:, i, j, k] = tf.cast(tf.random.categorical(prob, 1), tf.float32)[:, 0]
        return samples.numpy()

    @staticmethod
    def loss(y_hat, y):
        y_hat = tf.transpose(y_hat, [0, 4, 1, 2, 3])  # (B, H, W, C, K) -> (B, K, H, W, C)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

# Setting up the device for TensorFlow
device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"Using {device} device")

# Initialize model
pixel_cnn = PixelCNN(H=32, W=32, num_channels=1, num_colors=2)
sample_input = tf.cast(tf.random.uniform((10, 32, 32, 1), maxval=2, dtype=tf.int32), tf.float32)

orig_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=7, padding="same")
test_layer = MaskedConv2DA(in_channels=1, out_channels=64, kernel_size=7, padding="same")

output = pixel_cnn(sample_input)
# output = test_layer(sample_input)
print(output.shape)
