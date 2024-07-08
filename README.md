# Handwritten Digit Recognition Using AI

As part of the Introduction to Computer Science course at the University of Engineering and Technology (UTEC), I developed a system that uses artificial intelligence to recognize handwritten digits. The system allows users to upload images of handwritten numbers, which are then processed and classified using machine learning techniques. The key steps involved in the project are as follows:

	1.	Image Preprocessing: The uploaded image is first converted to grayscale using OpenCV. The image is then thresholded to create a binary image, and the bounding box of the handwritten digit is identified. The digit is cropped to this bounding box and resized to an 8x8 pixel image, which is a standard input size for digit recognition.
	2.	Inversion and Scaling: The pixel values of the preprocessed image are inverted (black to white and white to black) to match the format of the training dataset. The pixel values are then scaled to a range appropriate for the recognition algorithm.
	3.	Loading the Digits Dataset: The system uses the digits dataset from the sklearn library, which contains images of handwritten digits and their corresponding labels.
	4.	Distance Calculation: The Euclidean distance between the preprocessed input image and each image in the digits dataset is calculated. This distance measure helps in finding the most similar images in the dataset to the input image.
	5.	Result Identification: The indices of the three images with the smallest Euclidean distance to the input image are identified. These indices correspond to the most likely digits that the input image represents.
	6.	Displaying Results: The system prints the three most likely digits that match the input image. Additionally, it visualizes one of the images from the dataset to illustrate the recognition process.

This project demonstrates the application of image processing and machine learning techniques to solve a common problem of digit recognition, providing a foundation for more advanced AI applications in the future.