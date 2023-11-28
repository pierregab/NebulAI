import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import Angle
from astropy.io import fits
import astropy.units as u
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from astropy.utils.data import clear_download_cache


def load_and_preprocess_images_parallel(csv_file_path, image_size=512, fov_threshold=0.7/60, max_workers=5):
    """
    Load and preprocess images from a CSV file containing RA, DEC, and FOV using parallel processing.

    :param csv_file_path: Path to the CSV file.
    :param image_size: Size of the image to download.
    :param fov_threshold: Field of view threshold in degrees.
    :param max_workers: Maximum number of threads for parallel downloading.
    :return: Numpy array of preprocessed images.

    """
    # Clear the Astropy cache
    clear_download_cache()
    tasks = []
    with open(csv_file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            ra, dec, fov_str = row[2], row[3], row[4]
            fov = float(re.match(r'(\d+(\.\d+)?)', fov_str).group(0))/60.0

            if fov > fov_threshold:
                ra_deg, dec_deg = Angle(ra, unit=u.hourangle).degree, Angle(dec, unit=u.deg).degree
                url = construct_hips2fits_url(ra_deg, dec_deg, fov, image_size)
                tasks.append(url)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(download_and_normalize_image, tasks))

    return np.array(results)

def construct_hips2fits_url(ra, dec, fov, image_size):
    """
    Construct the request URL for the hips2fits service.

    :param ra: Right Ascension in degrees.
    :param dec: Declination in degrees.
    :param fov: Field of view in degrees.
    :param image_size: Size of the image in pixels.
    :return: URL string.
    """
    query_params = {
        'hips': 'DSS2 red',
        'ra': ra,
        'dec': dec,
        'fov': fov,
        'width': image_size,
        'height': image_size
    }
    return f'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?{urlencode(query_params)}'

def download_and_normalize_image(url):
    """
    Download and normalize an image from a URL.

    :param url: URL of the image to download.
    :return: Normalized image array.
    """
    hdul = fits.open(url)
    image = hdul[0].data
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def display_images(images, num_images=35, individual_image_index=16):
    """
    Display a set of images in a grid layout and one image individually.

    :param images: Array of images.
    :param num_images: Number of images to display in the grid.
    :param individual_image_index: Index of the image to display individually.
    """
    fig, axs = plt.subplots(7, 5, figsize=(10, 15))
    axs = axs.flatten()
    for i in range(num_images):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Object {i+1}')
    
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.imshow(images[individual_image_index], cmap='gray')
    ax2.axis('off')
    ax2.set_title(f'Object {individual_image_index + 1}')
    plt.show()

if __name__ == '__main__':
    images = load_and_preprocess_images_parallel('output33.csv')
    display_images(images)
    clear_download_cache()
