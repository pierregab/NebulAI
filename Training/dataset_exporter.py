import numpy as np
from PIL import Image
import os
import shutil

def clear_directory(directory):
    """
    Clear all files and subdirectories in the specified directory.

    :param directory: The directory to clear.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def create_directories(base_dir, sub_dirs):
    """
    Create base and subdirectories for image storage.

    :param base_dir: The base directory for storing images.
    :param sub_dirs: A list of subdirectories to create within the base directory.
    """
    os.makedirs(base_dir, exist_ok=True)
    for sub_dir in sub_dirs:
        sub_dir_path = os.path.join(base_dir, sub_dir)
        clear_directory(sub_dir_path)  # Clear the directory
        os.makedirs(sub_dir_path, exist_ok=True)

def split_dataset(images, not_images, train_ratio=0.8):
    """
    Split the images and not_images into train and test sets.

    :param images: Array of images.
    :param not_images: Array of not_images.
    :param train_ratio: The ratio of training set size to the total dataset size.
    :return: Train and test sets for images and not_images.
    """
    split_index_img = int(train_ratio * len(images))
    split_index_not_img = int(train_ratio * len(not_images))

    train_images = images[:split_index_img]
    test_images = images[split_index_img:]

    train_not_images = not_images[:split_index_not_img]
    test_not_images = not_images[split_index_not_img:]

    return train_images, test_images, train_not_images, test_not_images

def save_images(images, directory, prefix):
    """
    Save the images to the specified directory with a given prefix.

    :param images: Array of images.
    :param directory: Directory where images will be saved.
    :param prefix: Prefix for the saved image files.
    """
    for i, img in enumerate(images):
        img_path = os.path.join(directory, f"{prefix}_{i}.jpg")
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        pil_img.save(img_path)

def export_dataset(images, not_images, base_dir="image_dataset"):
    """
    Export the images and not_images datasets into train and test sets and save them.

    :param images: Array of images.
    :param not_images: Array of not_images.
    :param base_dir: The base directory for storing images.
    """
    # Create directories
    sub_dirs = ["train/images", "train/not_images", "test/images", "test/not_images"]
    create_directories(base_dir, sub_dirs)

    # Split dataset
    train_images, test_images, train_not_images, test_not_images = split_dataset(images, not_images)

    # Save images
    save_images(train_images, os.path.join(base_dir, "train/images"), "image")
    save_images(train_not_images, os.path.join(base_dir, "train/not_images"), "not_image")
    save_images(test_images, os.path.join(base_dir, "test/images"), "image")
    save_images(test_not_images, os.path.join(base_dir, "test/not_images"), "not_image")
