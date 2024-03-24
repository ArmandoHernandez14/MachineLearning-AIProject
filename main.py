import os
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = path_to_zip.replace('.zip', '')

# Setting up Image Data Generators
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=32,
  directory=train_dir,
  shuffle=True,
  target_size=(150, 150),
  class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=32,
  directory=validation_dir,
  target_size=(150, 150),
  class_mode='binary')

# Building the Model
model = models.Sequential([
  layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
  layers.MaxPooling2D(2,2),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Conv2D(128, (3,3), activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy'])

# Training the model 
history = model.fit(
  train_data_gen,
  steps_per_epoch=100,  # Number of batches to take from the generator per epoch
  epochs=15,
  validation_data=val_data_gen,
  validation_steps=50  # Number of batches to take from the validation generator per epoch
)

# Evaluating the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

print(f"Training accuracy: {acc[-1]}, Validation accuracy: {val_acc[-1]}")

