import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mping
from PIL import Image
import seaborn as sns
import os
import cv2
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from tensorflow.math import confusion_matrix

# The MNIST dataset is already pre-processed and returned as numpy arrays, so we don't need to convert it.

# Load the dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(type(X_train))

# Display the shape of the numpy arrays
print((X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))

# Visualizing the first image in the training set
plt.imshow(X_train[0], cmap='gray')  # The image is grayscale, so we use cmap='gray'
plt.show()

# Print the label for the 0th image
print(Y_train[0])
print(type(Y_train))  # It is also a numpy array
print(Y_train)

# Display the shapes of the label arrays
print(Y_train.shape, Y_test.shape)

# Display the unique values in the test and training labels
print(numpy.unique(Y_test))
print(numpy.unique(Y_train))
print(numpy.unique(X_test))
print(numpy.unique(X_train))

# Scale the pixel values to a range of 0-1
X_train = X_train / 255
X_test = X_test / 255
print(numpy.unique(X_test))
print(numpy.unique(X_test)) 

# BUILDING THE NEURAL NETWORK

# Setting up the layers of the neural network
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

# Compiling the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the neural network
model.fit(X_train, Y_train, epochs=10)

# Training accuracy is 99.0%

# Evaluate the model on test data
# loss, accuracy = model.evaluate(X_test, Y_test)
# print(accuracy)
# Test accuracy is 96.6%

# Visualizing the first image in the test set
plt.imshow(X_test[0], cmap='gray')
plt.show()

print(Y_test[0])

# Predicting the labels for the test set
Y_predictions = model.predict(X_test)
print(Y_predictions.shape)

# Explanation: Since there are 10,000 images in the test set, and the model predicts 10 values (one for each digit 0-9)
print(Y_predictions[0])

# Get the predicted label for the first image
label_prediction = numpy.argmax(Y_predictions[0])  # Find the index of the maximum value
print(label_prediction)  # 7 is the output

# Create a list of predicted labels for all test images
label_pred_for_input_features = [numpy.argmax(i) for i in Y_predictions]
print(label_pred_for_input_features)

# Y_test are the actual labels, and label_pred_for_input_features are the predicted ones

# Creating the confusion matrix
conf_mat = confusion_matrix(Y_test, label_pred_for_input_features)
print(conf_mat)

# Last part: Building a predictive model where we provide an image, and it predicts the value

# Saving images as PNG files
save_dir = "mnist_png"
os.makedirs(save_dir, exist_ok=True)
for i in range(len(X_train)):
    image = X_train[i]
    label = Y_train[i]

    # Save the image as a PNG file
    file_name = os.path.join(save_dir, f"mnist_{i}_label_{label}.png")
    plt.imsave(file_name, image, cmap='gray')

print("PNG images saved successfully.")

# The images are 28x28 pixels and grayscale, so no need to perform any additional tasks.

# Scaling is already done for the training data, so there's no need to repeat that.

# Read the image to be predicted
image_to_be_predicted_path = input("Enter the path of the image to be predicted: ")
image_to_be_predicted = cv2.imread(image_to_be_predicted_path, cv2.IMREAD_GRAYSCALE)

# Resize the image to 28x28 pixels
resized_image = cv2.resize(image_to_be_predicted, (28, 28))

# Display the image
cv2.imshow("Image to be Predicted", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Scale the image to the same range as the training data
scaled_img = resized_image / 255

# Predict the label for the image
input_prediction = model.predict(scaled_img.reshape(1, 28, 28))

# Get the predicted label
predicted_label = numpy.argmax(input_prediction)

print("Predicted Label:", predicted_label)
