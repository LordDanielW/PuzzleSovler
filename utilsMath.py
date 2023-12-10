import cv2
import numpy as np
from scipy import ndimage


def distance_squared_average(array1, array2, shift_range=3):
    array1 = np.array(array1)
    array2 = np.array(array2)
    min_length = min(len(array1), len(array2))

    min_distance = float("inf")
    best_shift = 0

    # Try shifting array2 within the range [-shift_range, shift_range]
    for shift in range(-shift_range, shift_range + 1):
        shifted_array2 = np.roll(array2, shift)
        truncated_array1 = array1[:min_length]
        truncated_array2 = shifted_array2[:min_length]
        squared_diff = np.square(truncated_array1 - truncated_array2)
        avg_squared_distance = np.mean(squared_diff)

        if avg_squared_distance < min_distance:
            min_distance = avg_squared_distance
            best_shift = shift

    return min_distance, best_shift


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


# Utility method to rotate a list of points
def rotate_points_list(points_list, width, height):
    # print(f"Type of points_list: {type(points_list)}")
    rotated_points = []
    for point in points_list:
        # Check the structure of the point and unpack x, y coordinates accordingly
        if isinstance(point, np.ndarray) and point.shape == (1, 2):
            x, y = point[0][0], point[0][1]
        elif isinstance(point, np.ndarray) and point.shape == (2,):
            x, y = point[0], point[1]
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = point
        else:
            print(point)
            raise TypeError("Unexpected point structure in points_list")

        # new_x = y
        # new_y = width - x
        new_x = height - y
        new_y = x
        rotated_points.append(np.array([[new_x, new_y]]))

        # print(f"Type of rotated_points: {type(rotated_points)}")

    return rotated_points
