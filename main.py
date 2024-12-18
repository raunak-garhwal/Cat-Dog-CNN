# Importing necessary libraries
import numpy as np  # For numerical operations on image data
from tensorflow.keras.models import Sequential  # For creating a sequential model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Layers for the CNN
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # For image augmentation
from tensorflow.keras.preprocessing import image  # For loading and preprocessing new images

# Step 1: Initializing the CNN
# Creating a sequential model to add layers one after another
classifier = Sequential()

# Step 2: Adding the first convolutional layer
# This layer applies 32 filters of size 3x3 to extract features from the input image
# 'relu' activation helps introduce non-linearity by replacing negative values with zero
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))

# Step 3: Adding the first pooling layer
# MaxPooling reduces the spatial dimensions (height and width) of the feature maps by a factor of 2
# This helps in reducing computation and avoiding overfitting
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 4: Adding a second convolutional layer
# This layer extracts more complex features from the feature maps generated by the first pooling layer
classifier.add(Conv2D(32, (3,3), activation='relu'))

# Step 5: Adding the second pooling layer
# Further reduces the spatial dimensions of the feature maps to focus on essential features
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 6: Flattening the feature maps
# Converts the 2D feature maps into a 1D vector to feed into the fully connected layers
classifier.add(Flatten())

# Step 7: Adding a fully connected layer
# This dense layer with 128 neurons processes the features extracted by the convolutional and pooling layers
# 'relu' activation introduces non-linearity to help model complex relationships
classifier.add(Dense(units=128, activation='relu'))

# Step 8: Adding the output layer
# Single neuron with 'sigmoid' activation for binary classification (cat/dog)
# Outputs a probability value between 0 and 1
classifier.add(Dense(units=1, activation='sigmoid'))

# Step 9: Compiling the CNN
# 'adam' is an efficient optimization algorithm for adjusting weights
# 'binary_crossentropy' is the loss function used for binary classification problems
# 'accuracy' metric is used to evaluate the model during training
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 10: Data Preprocessing
# Preparing image data for training and testing with data augmentation

# Image augmentation for the training set:
# - Rescale: Normalizes pixel values to the range [0, 1]
# - Shear: Randomly distorts the image
# - Zoom: Randomly zooms into the image
# - Horizontal flip: Randomly flips the image horizontally to increase variety
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Image preprocessing for the test set:
# Only rescaling to normalize pixel values to the range [0, 1]
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Preparing the training set
# Loads images from the training directory, applies augmentation, and resizes them to 64x64 pixels
# Images are loaded in batches of 32, and the labels are binary (0 or 1)
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# Preparing the test set
# Similar to the training set but without augmentation; used for validation
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary'
)

# Step 11: Training the CNN
# Fitting the model to the training data and validating it on the test data
# - steps_per_epoch: Number of batches per epoch (8000 training images / batch size of 32 = 250)
# - epochs: Number of times the entire dataset is passed through the model
# - validation_steps: Number of validation batches per epoch (2000 test images / batch size of 32 = 63)
classifier.fit(
    training_set,
    steps_per_epoch=250,
    epochs=15,
    validation_data=test_set,
    validation_steps=63
)

'''
# Step 12: Making new predictions
# Function to predict whether a new image is a cat or a dog
def make_prediction(image_path):
    # Load the image from the given path and resize it to match the input shape (64x64 pixels)
    test_image = image.load_img(image_path, target_size=(64,64))
    
    # Convert the image into a numpy array
    test_image = image.img_to_array(test_image)
    
    # Add an extra dimension to the array to match the input format of the model (batch size of 1)
    test_image = np.expand_dims(test_image, axis=0)

    # Predict the class of the image using the trained model
    # Returns a probability between 0 and 1
    result = classifier.predict(test_image)

    # Interpret the prediction: Class 1 = Dog, Class 0 = Cat
    if result[0][0] == 1:
        return 'Dog'
    else:
        return 'Cat'
'''

# Updated Step 12: Making new predictions
# Function to predict whether a new image is a cat, dog, or neither
def make_prediction(image_path, threshold=0.6):
    """
    Predicts if the image is a Cat, Dog, or Neither based on confidence.

    Parameters:
    - image_path: Path to the input image
    - threshold: Confidence threshold for prediction (default is 0.6)

    Returns:
    - String: 'Dog', 'Cat', or 'Neither Cat nor Dog'
    """
    # Load the image and preprocess
    test_image = image.load_img(image_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Get the prediction probability
    result = classifier.predict(test_image)[0][0]

    # Classify based on threshold and confidence
    if result > threshold:
        return 'Dog'
    elif result < (1 - threshold):
        return 'Cat'
    else:
        return 'Neither Cat nor Dog'

# Example Prediction
# Testing the prediction function with a sample image
# Replace 'test-image-1.png' with the actual path of the image to test
prediction = make_prediction('dataset/single_prediction/test-image-4.jpg')
print(f"\nThe image provided is of a '{prediction}'.")