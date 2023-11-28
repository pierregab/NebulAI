import numpy as np
from PIL import Image
import os
from tqdm import tqdm

def save_images(all_images, ra_list, dec_list, fov_list, base_dir="image_data_nebula", image_dir="images"):
    """
    Saves a list of images to a specified directory after emptying it.

    Args:
        all_images (list): A list of NumPy arrays representing images.
        ra_list (list): A list of right ascension values for each image.
        dec_list (list): A list of declination values for each image.
        fov_list (list): A list of field of view values for each image.
        base_dir (str): The base directory where the images will be saved.
        image_dir (str): The subdirectory where the images will be saved.
    """
    # Create directories
    os.makedirs(base_dir, exist_ok=True)
    full_image_dir = os.path.join(base_dir, image_dir)
    os.makedirs(full_image_dir, exist_ok=True)

    # Clear existing files in the directory
    for filename in os.listdir(full_image_dir):
        os.remove(os.path.join(full_image_dir, filename))

    # Save images
    for i, img in tqdm(enumerate(all_images), total=len(all_images), desc=f"Saving images to {image_dir}"):
        ra, dec, fov = ra_list[i], dec_list[i], fov_list[i]
        img_path = os.path.join(full_image_dir, f"image_{i}_{ra}_{dec}_{fov}.jpg")
        pil_img = Image.fromarray((img * 255).astype(np.uint8))
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        pil_img.save(img_path)

    print(f"All {len(all_images)} images saved to {image_dir}")

def main():
    # Example usage, assuming all_images, ra_list, dec_list, fov_list are defined
    #save_images(all_images, ra_list, dec_list, fov_list)
    return 0


if __name__ == '__main__':
    main()
