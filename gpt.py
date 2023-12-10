import cv2
import numpy as np

def extract_lines_from_hough(hough_lines, image_shape):
    # Step 1: Create accumulator array
    accumulator = np.zeros(image_shape, dtype=np.uint8)

    # Step 2: Perform Connected Component Labeling
    _, labeled_objects = cv2.connectedComponents(hough_lines)

    # Step 3: Extract centroids of each object
    centroids = []
    for label in range(1, np.max(labeled_objects) + 1):
        object_pixels = np.where(labeled_objects == label)
        centroid = (np.mean(object_pixels[1]), np.mean(object_pixels[0]))
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Step 4: Draw lines corresponding to centroids
    lines_image = np.zeros(image_shape, dtype=np.uint8)
    for centroid in centroids:
        rho, theta = centroid[1], centroid[0]
        a = np.cos(np.radians(theta))
        b = np.sin(np.radians(theta))
        x0 = int(a * rho)
        y0 = int(b * rho)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(lines_image, (x1, y1), (x2, y2), 255, 1)

    return lines_image

# Example usage
if __name__ == "__main__":
    # Assume hough_lines is the output of cv2.HoughLines
    # Create a binary image with the detected lines
    image_shape = (500, 500)  # Adjust to your image size
    hough_lines_image = np.zeros(image_shape, dtype=np.uint8)

    # Randomly place some lines in the Hough space
    for _ in range(10):
        rho = np.random.uniform(0, image_shape[0])
        theta = np.random.uniform(0, 180)
        a = np.cos(np.radians(theta))
        b = np.sin(np.radians(theta))
        x0 = int(a * rho)
        y0 = int(b * rho)
        x1 = int(x0 + 500 * (-b))
        y1 = int(y0 + 500 * (a))
        x2 = int(x0 - 500 * (-b))
        y2 = int(y0 - 500 * (a))
        cv2.line(hough_lines_image, (x1, y1), (x2, y2), 255, 1)

    # Extract lines based on centroids
    extracted_lines = extract_lines_from_hough(hough_lines_image, image_shape)

    # Display the results
    cv2.imshow('Original Hough Lines', hough_lines_image)
    cv2.imshow('Extracted Lines', extracted_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
