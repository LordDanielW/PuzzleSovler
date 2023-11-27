import cv2
import numpy as np


def find_bounding_box(image):
    # For grayscale image, we don't compare against a 3-channel color.
    # Instead, we simply look for pixels that are not black (i.e., not 0).
    pts = np.argwhere(image > 0)  # Assuming black pixels have value 0.

    # Find the minimum and maximum x and y coordinates.
    min_y, min_x = pts.min(axis=0)
    max_y, max_x = pts.max(axis=0)

    # Create a box around those points.
    # Adding a border of 5 pixels around the non-black area.
    min_y = max(min_y - 5, 0)
    min_x = max(min_x - 5, 0)
    max_y = min(max_y + 5, image.shape[0])
    max_x = min(max_x + 5, image.shape[1])

    # Return the coordinates of the box.
    return [(min_x, min_y), (max_x, max_y)]


def rotate_image_easy(image, angle):
    # Define the color of the border and the size of the border.
    background_color = (0, 0, 0)  # black
    border_size = max(image.shape)  # You may adjust this size as needed.

    # Create an expanded image by adding a border around the original image.
    expanded_image = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=background_color,
    )

    # Using OpenCV's built-in rotation function to rotate the image.
    # 'borderValue=background_color' fills the border with the specified color.
    # 'interpolation=cv2.INTER_LINEAR' is used here, but you can adjust as needed.
    rotated_image = cv2.rotate(expanded_image, angle)

    # Find the bounding box of the non-black areas.
    box = find_bounding_box(rotated_image)

    # Crop the image using the bounding box.
    cropped_image = rotated_image[box[0][1] : box[1][1], box[0][0] : box[1][0]]

    return cropped_image


def rotate_image(image, angle):
    # Define the color of the border and the size of the border.
    background_color = (0, 0, 0)  # black
    border_size = max(image.shape)  # You may adjust this size as needed.

    # Create an expanded image by adding a border around the original image.
    expanded_image = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=background_color,
    )

    # Using scipy's ndimage to rotate the image within the expanded border.
    # 'reshape=False' ensures the output shape is the same as the input image.
    # 'cval=0' sets the background to black where there are no image data.
    # 'order=1' (bilinear interpolation) is used here, but you can adjust as needed.
    rotated_image = ndimage.rotate(
        expanded_image, angle, reshape=False, cval=0, order=1
    )

    # Find the bounding box of the non-black areas.
    box = find_bounding_box(rotated_image)

    # Crop the image using the bounding box.
    cropped_image = rotated_image[box[0][1] : box[1][1], box[0][0] : box[1][0]]

    return cropped_image
