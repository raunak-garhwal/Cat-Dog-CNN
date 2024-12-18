# Importing necessary libraries
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image  # For loading and preprocessing new images
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Step 1: Initializing the CNN
classifier = Sequential()

# Step 2: Adding optimized convolutional and pooling layers
classifier.add(Conv2D(32, (3, 3), input_shape=(96, 96, 3), activation='relu'))  # Reduced input size to 96x96
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))  # Kept third layer lightweight
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3: Flattening
classifier.add(Flatten())

# Step 4: Fully connected layers
classifier.add(Dense(units=128, activation='relu'))  # Reduced Dense size
classifier.add(Dropout(0.4))  # Slightly increased Dropout
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.3))

# Output layer
classifier.add(Dense(units=1, activation='sigmoid'))

# Step 5: Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 6: Data Preprocessing with workers and optimization
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.1,  # Slightly reduced augmentation for faster processing
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(96, 96),  # Reduced image size to speed up training
    batch_size=32,
    class_mode='binary'
)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(96, 96),
    batch_size=32,
    class_mode='binary'
)

# # Step 7: Training with EarlyStopping and learning rate reduction
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# classifier.fit(
#     training_set,
#     steps_per_epoch=len(training_set),
#     epochs=15,  # Reduced to 15 epochs with early stopping
#     validation_data=test_set,
#     validation_steps=len(test_set),
#     callbacks=[early_stopping, reduce_lr],
#     workers=4,  # Multi-threaded data loading
#     use_multiprocessing=True  # Enable multiprocessing
# )

# Training with EarlyStopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

classifier.fit(
    training_set,
    steps_per_epoch=len(training_set),
    epochs=15,  # Reduced to 15 epochs with early stopping
    validation_data=test_set,
    validation_steps=len(test_set),
    callbacks=[early_stopping, reduce_lr]
)


# Step 8: Optimized Prediction Function
def make_prediction(image_path, threshold=0.6):
    test_image = image.load_img(image_path, target_size=(96, 96))  # Match new input size
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

    result = classifier.predict(test_image, verbose=0)[0][0]

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

# different approach to solve but also not working as expected
# lion image ---> predicting dog
