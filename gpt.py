import cv2
import numpy as np
# Read the input image
image = cv2.imread('tmp/test.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a GraphSegmentation object
segmentator = cv2.ximgproc.segmentation.createGraphSegmentation()

# Set the algorithm parameters (adjust as needed)
segmentator.setSigma(0.8)
segmentator.setK(300)
segmentator.setMinSize(100)

# Perform image segmentation
result = segmentator.processImage(gray)

# Display the segmented image
result = result.astype(np.uint8)
cv2.imshow('Segmented Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
