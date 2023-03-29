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
    cv2.imwrite("output_image.jpg", input_image)
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

    cv2.imwrite('output_image.jpg', input_image)
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

    cv2.imwrite('output_image.jpg', input_image)
    pass


def main():
    input_image = cv2.imread("flower.jpg")
    #level_1(input_image)
    #level_2(input_image)
    level_3(input_image)


if __name__ == "__main__":
    main()