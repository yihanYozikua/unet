import numpy as np
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.color import rgb2gray

# Load the model from the file
model = load_model('unet_membrane.keras')

# Simulate a single RGB image of shape (224, 224, 3)
single_image = np.random.rand(224, 224, 3)
# Resize the image to (256, 256)
resized_image = resize(single_image, (256, 256), mode='reflect', anti_aliasing=True)
# Convert the image to grayscale
grayscale_image = rgb2gray(resized_image)
# Expand dimensions to create a batch of size 1 and add a channel dimension
input_data = np.expand_dims(grayscale_image, axis=(0, -1))

# Predict using the model
predictions = model.predict(input_data, 30, verbose=1)

# Print the predictions
print(predictions)