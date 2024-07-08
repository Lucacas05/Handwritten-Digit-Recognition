import cv2
import numpy as np
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Threshold image
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    # Find bounding box
    x, y, w, h = cv2.boundingRect(image)
    # Crop image to bounding box
    image = image[y:y+h, x:x+w]
    # Resize image to 8x8
    image = cv2.resize(image, (8, 8))
    return image

# Load the image
img_array = cv2.imread("numbers/digits/IMG_1309.jpg", cv2.IMREAD_GRAYSCALE) ## Upload the image

# Preprocess the image
img_nueva = preprocess_image(img_array)

# Invert the pixel values
img_nueva = 255 - img_nueva

# Scale the pixel values
img_nueva = img_nueva * 16 / 255

# Load the example digits
digits = datasets.load_digits()

# Calculate the Euclidean distance between the reference image and all the images
distances = []
for image in digits["data"]:
    # Normalize both images
    img_nueva_norm = img_nueva.ravel() / np.linalg.norm(img_nueva.ravel())
    image_norm = image / np.linalg.norm(image)
    # Calculate distance
    diff = img_nueva_norm - image_norm
    distance = np.sqrt(np.sum(np.square(diff)))
    distances.append(distance)

# Get the indices of the three results with the smallest difference
indices = np.argsort(distances)[:3]

# Get the three results with the smallest difference
results = []
for index in indices:
    target = digits["target"][index]
    results.append(target)

print(results)

digits = datasets.load_digits()
image = digits["images"][1796]  # Obtén la imagen de interés

# Configuración de la figura
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xticks([])  # Oculta las marcas en el eje x
ax.set_yticks([])  # Oculta las marcas en el eje y

# Mostrar los números en la matriz con colores
ax.imshow(image, cmap="viridis")

plt.show()
