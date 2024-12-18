# Importing necessary libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image  # For loading and preprocessing new images

# Step 1: Initializing the CNN
classifier = Sequential()

# Step 2: Adding the first convolutional and pooling layers
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
classifier.add(BatchNormalization())  # Normalize after convolution
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3: Adding a second convolutional and pooling layer with more filters
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 4: Adding a third convolutional and pooling layer
classifier.add(Conv2D(128, (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 5: Flattening the feature maps
classifier.add(Flatten())

# Step 6: Fully connected layers
classifier.add(Dense(units=256, activation='relu'))  # Increased neurons
classifier.add(Dropout(0.5))  # Dropout to prevent overfitting
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))

# Step 7: Output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Step 8: Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 9: Data Preprocessing with larger image size
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Training and test set preparation
training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(128, 128),  # Increased image size
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Step 10: Training the CNN with more epochs
classifier.fit(
    training_set,
    steps_per_epoch=250,  # 8000 images / 32 batch size
    epochs=20,  # Increased to 20 epochs
    validation_data=test_set,
    validation_steps=63
)

# Step 11: Making Predictions with updated threshold logic
def make_prediction(image_path, threshold=0.6):
    """
    Predicts if the image is a Cat, Dog, or Neither based on confidence.

    Parameters:
    - image_path: Path to the input image
    - threshold: Confidence threshold for prediction (default is 0.6)

    Returns:
    - String: 'Dog', 'Cat', or 'Neither Cat nor Dog'
    """
    # Load and preprocess the image
    test_image = image.load_img(image_path, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    # Get the prediction probability
    result = classifier.predict(test_image)[0][0]

    # Classify based on threshold
    if result > threshold:
        return 'Dog'
    elif result < (1 - threshold):
        return 'Cat'
    else:
        return 'Neither Cat nor Dog'

# Example prediction
image_path = 'dataset/single_prediction/test-image-4.jpg'
prediction = make_prediction(image_path)
print(f"\nThe image provided is of a '{prediction}'.")


# taking too much time and resources to run
