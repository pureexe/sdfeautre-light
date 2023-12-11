import torch
import numpy as np
from PIL import Image
import skimage
import cv2



def pil_square_image(image, desired_size = (512,512), interpolation=Image.LANCZOS):
    """
    Make top-bottom border
    """
    # Don't resize if already desired size (Avoid aliasing problem)
    if image.size == desired_size:
        return image
    
    # Calculate the scale factor
    scale_factor = min(desired_size[0] / image.width, desired_size[1] / image.height)

    # Resize the image
    resized_image = image.resize((int(image.width * scale_factor), int(image.height * scale_factor)), interpolation)

    # Create a new blank image with the desired size and black border
    new_image = Image.new("RGB", desired_size, color=(0, 0, 0))

    # Paste the resized image onto the new image, centered
    new_image.paste(resized_image, ((desired_size[0] - resized_image.width) // 2, (desired_size[1] - resized_image.height) // 2))
    
    return new_image