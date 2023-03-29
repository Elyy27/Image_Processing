import cv2
import random
import numpy as np


def level_1(input_image):
    # Define the number of circles to draw and their properties
    num_circles = 3
    circle_radius = random.randint(10, 20)
    circle_thickness = -1

    # Draw the circles in random positions
    for i in range(num_circles):
        # Generate random coordinates for the center of the circle
        center_x = random.randint(circle_radius, input_image.shape[1] - circle_radius)
        center_y = random.randint(circle_radius, input_image.shape[0] - circle_radius)
        center = (center_x, center_y)

        circle_color = (np.random.uniform(0, 255), np.random.uniform(0, 255), np.random.uniform(0, 255))
        # Draw the circle
        cv2.circle(input_image, center, circle_radius, circle_color, circle_thickness)

    # Save the modified image as output
    cv2.imwrite("output_image_level1.jpg", input_image)

    pass

def level_2(input_image):
    img = input_image.copy()
    img_canny = cv2.Canny(img, 150, 200)
    MAX_CHANGES = 5
    contours, _ = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        area = w * h
        rand = np.random.randint(1, 100)
        if area >= 100 and area <= 1000 and rand % 3 == 0:
            new_color = (np.random.uniform(0, 255), np.random.uniform(0, 255), np.random.uniform(0, 255))
            input_image[y:y + h, x:x + w] = new_color
            MAX_CHANGES -= 1
            if MAX_CHANGES == 0:
                break

    cv2.imwrite('output_image_level2.jpg', input_image)
    pass


def level_3(input_image):
    img = input_image.copy()
    img_canny = cv2.Canny(img, 150, 200)
    MAX_CHANGES = 5
    contours = [(c, cv2.contourArea(c)) for c in cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]]
    for contour in contours:
        area = contour[1]
        rand = np.random.randint(1, 100)
        if area >= 500 and area <= 3000 and rand % 3 == 0:
            new_color = (np.random.uniform(0, 255), np.random.uniform(0, 255), np.random.uniform(0, 255))
            cv2.fillPoly(input_image, [contour[0]], new_color)
            MAX_CHANGES -= 1
            if MAX_CHANGES == 0:
                break

    cv2.imwrite('output_image_level3.jpg', input_image)
    pass

def level_4(input_image):
    img = input_image.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, 150, 200)

    # Find contours
    contours, _ = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Find bounding box of largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x = cx - size // 2
    y = cy - size // 2
    w, h = size, size

    # Extract portion of image within bounding box
    img_roi = img[y:y + h, x:x + w]

    # Rotate the image portion by specified angle
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 90, 1)
    rotated_img = cv2.warpAffine(img_roi, rotation_matrix, (w, h))

    # Replace the rotated portion in original image
    img[y:y + h, x:x + w] = rotated_img

    # Save the output image
    cv2.imwrite('output_image_level4.jpg', img)

def level_5(input_image):
    img = cv2.imread('input_image.jpg')

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    canny = cv2.Canny(gray, 100, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Find bounding box of largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Create a mask with the same size as the image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Draw the largest contour on the mask in white color
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)

    # Invert the mask
    mask = cv2.bitwise_not(mask)

    # Set the pixels inside the contour to black color
    color = img[y, x-5]
    img[mask == 0] = (color)
    cv2.imwrite('output_image_level5.jpg', img)

def dif(img1, img2):
    diff = cv2.absdiff(img1, img2)

    # Convert the difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours in the binary imageS
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 2)

    diff = cv2.absdiff(img1, img2)

    # Convert the difference image to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to create a binary image
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to remove noise and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 2)

    cv2.imwrite('dif_level5.jpg', img1)

def main():
    input_image = cv2.imread("input_image.jpg")
    output_image = cv2.imread("output_image_level5.jpg")
    #level_1(input_image)
    #level_2(input_image)
    #level_3(input_image)
    #level_4(input_image)
    #level_5(input_image)
    dif(output_image, input_image)

if __name__ == "__main__":
    main()