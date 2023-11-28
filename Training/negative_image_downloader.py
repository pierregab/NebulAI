import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u
from urllib.parse import urlencode
import concurrent.futures
from astropy.utils.data import clear_download_cache
from tqdm import tqdm

def generate_random_coordinates(num_positions, fov_range):
    """
    Generate random RA and Dec coordinates and FOV values.

    :param num_positions: Number of random positions to generate.
    :param fov_range: Range of FOV values in degrees.
    :return: Lists of RA, Dec, and FOV values.
    """
    ra = np.random.uniform(low=0, high=360, size=num_positions)
    dec = np.random.uniform(low=-90, high=90, size=num_positions)
    fov_choices = np.random.choice(fov_range, size=num_positions)
    return ra, dec, fov_choices

def download_and_process_image(ra_deg, dec_deg, fov, image_size=512):
    """
    Download and process a single image from given RA, Dec, and FOV.

    :param ra_deg: Right Ascension in degrees.
    :param dec_deg: Declination in degrees.
    :param fov: Field of view in degrees.
    :param image_size: Size of the image in pixels.
    :return: Processed image array.
    """
    query_params = {
        'hips': 'DSS2 red',
        'ra': ra_deg,
        'dec': dec_deg,
        'fov': fov,
        'width': image_size,
        'height': image_size
    }
    url = f'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?{urlencode(query_params)}'
    hdul = fits.open(url, cache=False, show_progress=False)  # Added arguments to suppress messages
    image = hdul[0].data
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def download_negative_samples(num_samples=200, fov_range=np.arange(0.7, 20, 0.1)/60):
    """
    Download a specified number of negative sample images.

    :param num_samples: Number of negative samples to download.
    :param fov_range: Range of FOV values in degrees.
    :return: Numpy array of negative sample images.
    """
    ra, dec, fov_choices = generate_random_coordinates(num_samples, fov_range)
    images = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Create a list to hold the future results
        futures = [executor.submit(download_and_process_image, ra[i], dec[i], fov_choices[i]) for i in range(num_samples)]
        
        # Use tqdm to create a progress bar for the as_completed iterator
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_samples):
            images.append(future.result())

    return np.array(images)

def display_images(images, num_images=35):
    """
    Display a set of images in a grid layout.

    :param images: Array of images.
    :param num_images: Number of images to display.
    """
    fig, axs = plt.subplots(7, 5, figsize=(10, 15))
    axs = axs.flatten()
    for i in range(num_images):
        axs[i].imshow(images[i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Object {i+1}')
    plt.show()

def clear_astropy_cache():
    """
    Clear the download cache used by Astropy.
    """
    clear_download_cache()


if __name__ == '__main__':
    not_images = download_negative_samples()
    display_images(not_images)
    clear_astropy_cache()

