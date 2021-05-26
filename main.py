import numpy as np 
import tensorflow as tf
from tensorflow import keras 

model = tf.keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1.0, 9.0, 2.0, 12.0, 32.0, 45.0], dtype=float)
ys = np.array([2.0, 18.0, 4.0, 24.0, 64.0, 90.0], dtype=float)

model.fit(xs, ys, epochs=10)

print(model.predict([90.0], batch_size=1))

# y = 9x + 2


