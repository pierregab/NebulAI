import os
import sqlite3
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import requests
from PIL import Image, ImageDraw
from io import BytesIO
from astropy.visualization import ZScaleInterval
import random
import json

# Define paths
metadata_db = 'iphas-images.sqlite'  # Path to the IPHAS SQLite metadata file
input_csv = 'HASH.csv'  # Path to the input CSV file
output_dir = 'output_images'  # Directory to save the PNG images

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Define the fixed FOV in degrees (107 arcseconds)
fixed_fov_deg = 107 / 3600.0  # Convert arcseconds to degrees

def convert_coords(ra_str, dec_str):
    try:
        sky_coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
        return sky_coord.ra.degree, sky_coord.dec.degree
    except Exception as e:
        print(f"Error converting coordinates RA: {ra_str}, DEC: {dec_str}: {e}")
        return np.nan, np.nan

def convert_size(size):
    try:
        if pd.isna(size):
            return np.nan
        if isinstance(size, str):
            size = ''.join(filter(lambda x: x.isdigit() or x == '.', size))
            major_size = float(size.split('.')[0] + '.' + ''.join(size.split('.')[1:]))  # Handle multiple decimal points
        else:
            major_size = size
        return major_size / 3600.0  # Convert arcseconds to degrees
    except Exception as e:
        print(f"Error converting size: {size}: {e}")
        return np.nan

def read_input_csv(file_path):
    df = pd.read_csv(file_path)
    df = df[['Name', 'RAJ2000', 'DECJ2000', 'MajDiam']]  # Adjust based on your CSV columns
    df[['RA', 'DEC']] = df.apply(lambda row: convert_coords(row['RAJ2000'], row['DECJ2000']), axis=1, result_type='expand')
    df['Size'] = df['MajDiam'].apply(convert_size)
    df = df.dropna(subset=['RA', 'DEC', 'Size'])
    return df

def find_iphas_images_for_coords(conn, ra, dec):
    query = f"""
    SELECT run, ccd, url, band, qcgrade FROM images
    WHERE ra_min <= {ra} AND ra_max >= {ra}
    AND dec_min <= {dec} AND dec_max >= {dec}
    AND qcgrade = 'A'
    """
    result = pd.read_sql_query(query, conn)
    return result

def draw_bbox(img, bbox):
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline='red', width=2)
    return img

def download_and_crop_image(run, ccd, ra, dec, size_deg, annotations, debug=False):
    url = f"http://www.iphas.org/data/images/r{str(run)[:3]}/r{run}-{ccd}.fits.fz"
    try:
        response = requests.get(url)
        response.raise_for_status()

        hdul = fits.open(BytesIO(response.content))

        for hdu in hdul:
            header = hdu.header
            if 'SECPPIX' in header:
                wcs = WCS(header)
                position = SkyCoord(ra, dec, unit="deg")
                pixel_scale = header['SECPPIX'] / 3600.0  # Convert arcseconds to degrees
                crop_size_pixels = int(fixed_fov_deg / pixel_scale)

                try:
                    cutout = Cutout2D(hdu.data, position, (crop_size_pixels, crop_size_pixels), wcs=wcs)
                    
                    # Convert the original position to pixel coordinates
                    original_position_pixel = wcs.world_to_pixel(position)
                    original_x, original_y = original_position_pixel

                    # Apply random shift to the position
                    nebula_size_pixels = int(size_deg / pixel_scale * 512 / crop_size_pixels)
                    max_shift = (512 - nebula_size_pixels) // 2
                    shift_x = random.randint(-max_shift, max_shift)
                    shift_y = random.randint(-max_shift, max_shift)

                    shifted_x = original_x + shift_x / (512 / crop_size_pixels)
                    shifted_y = original_y + shift_y / (512 / crop_size_pixels)

                    shifted_position = wcs.pixel_to_world(shifted_x, shifted_y)

                    # Create a new cutout with the shifted position
                    cutout = Cutout2D(hdu.data, shifted_position, (crop_size_pixels, crop_size_pixels), wcs=wcs)

                    # Apply a less aggressive ZScale normalization
                    interval = ZScaleInterval(contrast=0.1, max_reject=0.3, krej=1.5)  # Adjusted parameters
                    vmin, vmax = interval.get_limits(cutout.data)
                    img_data = np.clip((cutout.data - vmin) / (vmax - vmin) * 255.0, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_data).convert("RGB")
                    img = img.resize((512, 512), Image.LANCZOS)  # Resize to 512x512

                    # Bounding box coordinates
                    half_size = nebula_size_pixels // 2
                    center_x = 512 // 2 - shift_x
                    center_y = 512 // 2 - shift_y
                    bbox = [center_x - half_size, center_y - half_size, center_x + half_size, center_y + half_size]

                    if debug:
                        print(f"Drawing bounding box for {run}-{ccd} with shift ({shift_x}, {shift_y})")
                        img = draw_bbox(img, bbox)

                    png_file_path = os.path.join(output_dir, f'{run}-{ccd}_shifted_512x512.png')
                    img.save(png_file_path)
                    print(f"Image saved to {png_file_path}")

                    # Save annotation
                    annotation = {
                        'file_path': png_file_path,
                        'bbox': bbox,
                        'class': 'nebula'  # Adjust based on your classes
                    }
                    annotations.append(annotation)
                except Exception as e:
                    print(f"Cutout failed for {run}-{ccd} at RA: {ra}, DEC: {dec} with size {size_deg} deg: {e}")

                return

        print(f"Skipping image {url} due to missing SECPPIX keyword")
    except Exception as e:
        print(f"Error downloading or processing image {url}: {e}")

def list_iphas_images_for_pn(pn_data, conn, debug=False):
    all_results = []
    annotations = []

    for index, row in pn_data.iterrows():
        ra = row['RA']
        dec = row['DEC']
        size_deg = row['Size']
        pn_name = row['Name']

        image_data = find_iphas_images_for_coords(conn, ra, dec)
        if not image_data.empty:
            image_data = image_data[(image_data['band'].str.lower() == 'halpha')]
            if not image_data.empty:
                # Select the best quality image
                image_data = image_data.head(1)  # Select the best quality image
                
                image_data['pn_index'] = index
                image_data['pn_name'] = pn_name
                image_data['pn_ra'] = ra
                image_data['pn_dec'] = dec
                image_data['pn_size'] = size_deg  # Use fixed size for all images
                image_data['num_panels'] = len(image_data)
                all_results.append(image_data)

                for _, img_row in image_data.iterrows():
                    download_and_crop_image(img_row['run'], img_row['ccd'], ra, dec, size_deg, annotations, debug)

    if all_results:
        all_results_df = pd.concat(all_results, ignore_index=True)
        all_results_df.to_csv('pn_iphas_images.csv', index=False)
        print("Saved the list of possible IPHAS images to 'pn_iphas_images.csv'.")
    else:
        print("No matching IPHAS images found for any planetary nebula.")

    # Save annotations to a JSON file
    with open('annotations.json', 'w') as f:
        json.dump(annotations, f, indent=4)
    print("Saved annotations to 'annotations.json'.")

def main():
    conn = sqlite3.connect(metadata_db)
    conn.row_factory = sqlite3.Row  # To return rows as dictionaries

    pn_data = read_input_csv(input_csv)
    if pn_data.empty:
        print("No PN data found. Exiting.")
        return

    print(f"Queried {len(pn_data)} planetary nebulae from the input CSV.")
    
    debug = False  # Set to True to enable debugging output with bounding boxes
    list_iphas_images_for_pn(pn_data, conn, debug)

    conn.close()
    print("Process completed.")

if __name__ == '__main__':
    main()
