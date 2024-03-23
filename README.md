# MachineLearning-AIProject

- Programming languages and libaries used: Python and tensorflow

- AI can differentiate between animals and classify them to their corresponding categories.
    - This model was trained to be able to differentiate between a cat and a dog.

- Model training is done in main.py with data found on https://www.kaggle.com/datasets/tongpython/cat-and-dog/data

Steps to Approach the problem 
1) Prepare the dataset
    - We will need a dataset of dog images and a dataset of cat images
2) Setting up Image Data Generators
    - We will use "ImageDataGenerator" to load our images in batches, and apply data augmentation to diversify our training dataset
3) Building the model
    - Here we define a simple Convolutional Neural Network (CNN) that is suitable for image classification tasks
4) Compiling the model
5) Training the model
    - Feed the training data to the model. 
    - The model learns to associate images and labels.
    - You ask the model to make predictions about a test set—in this example, the test_images array.
    - Verify that the predictions match the labels from the test_labels array.
6) Evaluating the model