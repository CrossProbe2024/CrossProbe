import tensorflow as tf
from tensorflow import keras

keras.mixed_precision.set_global_policy("mixed_float16")
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(10)(inputs)
outputs = keras.layers.Activation("softmax", dtype="float32")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.RMSprop()
)

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype("float32") / 255

_ = model.fit(x_train, y_train, batch_size=128, epochs=1, steps_per_epoch=1, verbose=0)
