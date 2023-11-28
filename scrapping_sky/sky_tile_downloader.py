import numpy as np
import requests
from astropy.io import fits
from astropy.utils.data import clear_download_cache
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
    fig, ax = plt.subplots()
    rect = Rectangle((ra_min, dec_min), ra_max-ra_min, dec_max-dec_min, fill=False, edgecolor='red')
    ax.add_patch(rect)
    ax.set_xlim(ra_min, ra_max)
    ax.set_ylim(dec_min, dec_max)
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
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
