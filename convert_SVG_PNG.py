import os
import re
import cairosvg
import cv2
import numpy as np

# Set your input and output directories
input_dir = "Puzzles/SVG"
output_dir = "Puzzles/Original"


def sanitize_filename(filename):
    """
    Remove special characters from a filename, except for the dot ('.').

    :param filename: Original filename
    :return: Sanitized filename
    """
    # First, remove the file extension (if present) so it doesn't get affected by the character removal.
    file_base, file_extension = os.path.splitext(filename)

    # Remove or replace non-alphanumeric characters (excluding the period) with an empty string.
    sanitized_base = re.sub(r"[^A-Za-z0-9]+", "", file_base)

    # Reattach the file extension and return the sanitized filename.
    return f"{sanitized_base}{file_extension}"


def svg_to_png(input_directory, output_directory):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".svg"):
            # Construct full (absolute) paths for input SVG and output PNG
            svg_file_path = os.path.join(input_directory, filename)
            sanitized_filename = sanitize_filename(filename)
            png_file_name = f"{os.path.splitext(sanitized_filename)[0]}.png"
            png_temp_path = os.path.join(output_directory, f"temp_{png_file_name}")
            png_file_path = os.path.join(output_directory, png_file_name)

            # Convert the SVG file to a temporary PNG file
            try:
                cairosvg.svg2png(url=svg_file_path, write_to=png_temp_path)
            except Exception as e:
                print(f"An error occurred while converting {filename}: {e}")
                continue

            # Read the temporary PNG file and remove the alpha channel
            img = cv2.imread(png_temp_path, cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:  # Check for alpha channel
                # Create a 3-channel white background
                background = (
                    np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
                )

                # Split image channels and merge back without alpha
                b, g, r, a = cv2.split(img)
                img = cv2.merge((b, g, r))

                # Use alpha channel as mask to combine the image with the background
                mask = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) / 255.0
                img = (img * mask + background * (1 - mask)).astype(np.uint8)

            # Add 1px black border
            bordered_img = np.zeros(
                (img.shape[0] + 2, img.shape[1] + 2, 3), dtype=np.uint8
            )
            bordered_img[1:-1, 1:-1] = img

            # Save the image
            cv2.imwrite(png_file_path, bordered_img)

            # Remove the temporary PNG file
            os.remove(png_temp_path)

            print(
                f"Converted and saved {filename} as {png_file_name} without the alpha channel."
            )


# Convert all SVG files in the input directory to PNG format in the output directory
svg_to_png(input_dir, output_dir)
