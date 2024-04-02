# Allows us to interact with the operating system
import os
# TensorFlow (deep learning library) will be used for model building, training, and evaluation
import tensorflow as tf
# Keras is a high-level neural networks API, running on top of TensorFlow
# It simplifies many tasks in creating deep learning models
from keras import layers, models
# Imports the ImageDataGenerator class from TensorFlow's Keras API, used for data augmentation and preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Getting Data
  # Dataset (cats and dogs images) stored in a zip file
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
  # Specifies the local path of the zip file
path_to_zip = "/Users/hxzheng/Downloads/cats_and_dogs_filtered"
  # PATH indicats the directory where the extracted files are located
PATH = path_to_zip.replace('.zip', '')



# Setting up Image Data Generators
  # The training and validation directories are set by joining the PATH with subdirectories 'train' and 'validation'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

  # Rescale the image pixel values from the range [0,255] to [0,1] for neural network input
train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

  # Load images from their respective directories
    # Applying the specified transformations (rescaling) 
    # Set parameters such as batch size, image size, shuffle mode, and class mode (binary for two classes).
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
  # Convolutional Layers (Conv2D): Perform the convolution operation, extracting features from the input images
    # Parameters include the number of filters, kernel size, activation function, and input shape for the first layer
  # Pooling Layers (MaxPooling2D): Reduce the spatial dimensions (height, width) of the input volume for the next convolutional layer
    # help with reducing overfitting and computational load
model = models.Sequential([
  layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
  layers.MaxPooling2D(2,2),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Conv2D(128, (3,3), activation='relu'),
  layers.MaxPooling2D(2,2),
  # Flattening (Flatten): Converts the 2D matrix data to a vector so it can be fed into the dense layers
  layers.Flatten(),
  # Dense Layers: Fully connected layers that perform classification on the features extracted by the convolutional and pooling layers
    # The last dense layer uses a sigmoid activation function for binary classification.
  layers.Dense(512, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])



# Compiling the model
  # The model is compiled with: 
    # The Adam optimizer
    # Binary cross-entropy loss function (suitable for binary classification tasks)
    # Accuracy as the metric for evaluation
model.compile(optimizer='adam',
  loss='binary_crossentropy',
  metrics=['accuracy'])

# Training the model 
  # The model is trained using the model.fit method
  # The training data generator specifies:
    # Steps per epoch, number of epochs, validation data generator, and validation steps per epoch.
history = model.fit(
  train_data_gen,
  steps_per_epoch=100,  # Number of batches to take from the generator per epoch
  epochs=15,
  validation_data=val_data_gen,
  validation_steps=50  # Number of batches to take from the validation generator per epoch
)



# Evaluating the model
  # After training, the training and validation accuracies are extracted from the history object and printed out
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

print(f"Training accuracy: {acc[-1]}, Validation accuracy: {val_acc[-1]}")



# Save the model 
  # The trained model is saved to the file 'my_model.keras' using the model.save method for later use or further evaluation.
model.save('my_model.keras')

