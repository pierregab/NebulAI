import numpy as np
import requests
from astropy.io import fits
from astropy.utils.data import clear_download_cache
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import astropy.units as u
from matplotlib.patches import Polygon

def ra_molweid_to_degree(ra_molweid):
    return -ra_molweid if ra_molweid < 0 else 360 - ra_molweid

def download_and_process_image(params):
    ra_center, dec_center, fov_size = params
    largeur, hauteur = 512, 512
    ra_center_d = ra_molweid_to_degree(ra_center)

    query_params = {
        'hips': 'DSS2 red',  # or 'DSS2 blue' for blue filter
        'ra': ra_center_d,
        'dec': dec_center,
        'fov': fov_size,
        'width': largeur,
        'height': hauteur
    }
    url = f'http://alasky.u-strasbg.fr/hips-image-services/hips2fits?{urlencode(query_params)}'
    hdul = fits.open(url)
    image = hdul[0].data
    # Normalize and process the image as needed
    return image

def download_sky_region(ra_min, ra_max, dec_min, dec_max, fov_step=5/60):
    # Define ranges and initialize lists
    ra_range = np.arange(ra_min, ra_max, fov_step)
    dec_range = np.arange(dec_min, dec_max, fov_step)
    params_list = [(ra, dec, fov_step) for ra in ra_range for dec in dec_range]

    # Download images in parallel
    images = []
    with ThreadPoolExecutor() as executor:
        futures = list(tqdm(executor.map(download_and_process_image, params_list), total=len(params_list)))
        images.extend(futures)

    return images

def plot_sky_region(ra_min, ra_max, dec_min, dec_max):
    # Convert the RA and DEC limits to radians for Mollweide projection
    ra_min_rad, ra_max_rad = np.radians([ra_min, ra_max]) - np.pi
    dec_min_rad, dec_max_rad = np.radians([dec_min, dec_max])

    # Define the vertices of the rectangle in the Mollweide projection
    vertices = [
        (ra_min_rad, dec_min_rad), (ra_min_rad, dec_max_rad),
        (ra_max_rad, dec_max_rad), (ra_max_rad, dec_min_rad),
        (ra_min_rad, dec_min_rad)
    ]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='mollweide')
    ax.grid(True)

    # Create a polygon patch and add it to the axis
    sky_region = Polygon(vertices, fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(sky_region)

    # Set labels for RA and DEC
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.set_xticklabels(['14h', '16h', '18h', '20h', '22h', '0h', '2h', '4h', '6h', '8h', '10h'])
    ax.set_title('Sky Region')

    plt.show()

def clear_astropy_cache():
    clear_download_cache()

def main():

    ra_min, ra_max, dec_min, dec_max = 75, 80, 0, 5
    images = download_sky_region(ra_min, ra_max, dec_min, dec_max)
    plot_sky_region(ra_min, ra_max, dec_min, dec_max)

    clear_astropy_cache()

    # Additional processing can be added here

if __name__ == '__main__':
    main()
