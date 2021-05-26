import numpy as np 
import tensorflow as tf
from tensorflow import keras 

model = tf.keras.Sequential(
    [keras.layers.Dense(units=1, input_shape=[1])]
    
    )

model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([11.0, 20.0, 29.0, 38.0, 47.0, 56.0], dtype=float)

model.fit(xs, ys, epochs=10)

print(model.predict([1.0], batch_size=1))
# y = 9x + 2
