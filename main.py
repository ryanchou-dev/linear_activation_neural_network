import tensorflow as tf 
import numpy as np 
from tensorflow import keras 

model = tf.keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])]
    )

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1.0, 4.0, 3.0, 6.0, 23.0, 19.0], dtype=float)
ys = np.array([11.0, 38.0, 4.0, 7.0, 24.0, 20.0], dtype=float)

model.fit(xs, ys, epochs=100)

print(model.predict([10.0]))

# y = 9x + 2


