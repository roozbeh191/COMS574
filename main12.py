from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense

image_dir = 'C:/Users/reghb/OneDrive/Desktop/MATALB code/Machine learning/Project- COMS574/images9.30%/'
images = []
for filename in os.listdir(image_dir):
    if filename.endswith('.png'):
        image = Image.open(os.path.join(image_dir, filename))
        image_data = np.asarray(image)
        images.append(image_data)
        
x_data = np.array(images)
y_data = np.loadtxt('stiffness7.30%.txt')
y_data = y_data/max(y_data)
y_data = y_data.reshape((-1, 1))

#=============================================================================
x_train=x_data[0:360,:,:]
y_train=y_data[0:360]
x_test=x_data[360:,:]
y_test=y_data[360:]
#=============================================================================

model = Sequential([
    Conv2D(100, (3,3), strides=(2, 2), activation='relu', input_shape=(150, 150, 1)),
    BatchNormalization(),
    Conv2D(200, (3,3), strides=(2, 2), activation='relu'),
    Conv2D(300, (3,3), strides=(2, 2), activation='relu'),
    BatchNormalization(),
    Conv2D(400, (3,3), strides=(2, 2), activation='relu'),
    Conv2D(500, (3,3), strides=(2, 2), activation='relu'),
    BatchNormalization(),
    Flatten(),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(100, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

model.summary()
#=============================================================================

model.compile(optimizer='adam', loss='mse', metrics=['mse'])
tf.keras.optimizers.Adam(learning_rate=0.0005)
history = model.fit(x_train, y_train, epochs=50, batch_size=20, validation_split=0.15)

#=============================================================================

test_loss= model.evaluate(x_test, y_test)
y_test_pred = model.predict(x_test)

#=============================================================================

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss (MSE)')
plt.xlabel('epoch')
plt.ylim([0, 0.4])
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

#=============================================================================
