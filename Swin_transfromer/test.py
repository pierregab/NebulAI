import pandas as pd
import requests
from bs4 import BeautifulSoup
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np

# Define paths
input_csv = 'input_nebulae.csv'  # Path to the input CSV file
output_csv = 'combined_nebulae_catalog.csv'  # Path to the output CSV file

def fetch_classical_catalog():
    url = "http://cdsarc.u-strasbg.fr/viz-bin/nph-Cat/html?VII/219"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Parse the table from the HTML response
    table = soup.find('table')
    rows = table.find_all('tr')

    # Extract data from the table
    data = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 5:
            # Extract relevant data
            name = cols[1].text.strip()
            ra_str = cols[2].text.strip()
            dec_str = cols[3].text.strip()
            size_str = cols[5].text.strip().split('x')[0]

            # Combine RA and DEC if RA is missing
            if not ra_str:
                ra_dec_str = dec_str
            else:
                ra_dec_str = ra_str + " " + dec_str

            # Format and convert RA and DEC
            ra_deg, dec_deg = convert_coords(ra_dec_str)

            # Convert size
            size_deg = convert_size(size_str)

            # Append data to list
            data.append([name, ra_deg, dec_deg, size_deg])

    return pd.DataFrame(data, columns=['Name', 'RA', 'DEC', 'Size'])

def convert_coords(ra_dec_str):
    """Converts combined RA and DEC strings to decimal degrees."""
    try:
        # Clean up the string
        ra_dec_str = ra_dec_str.replace(' +', '+').replace(' -', '-').replace(' ', ':')
        
        # Split RA and DEC based on the presence of '+' or '-'
        for i, char in enumerate(ra_dec_str):
            if char in ['+', '-']:
                ra_str = ra_dec_str[:i].strip()
                dec_str = ra_dec_str[i:].strip()
                break

        # Replace multiple colons with a single colon
        ra_str = ':'.join(ra_str.split(':'))
        dec_str = ':'.join(dec_str.split(':'))

        # Parse RA and DEC using SkyCoord
        sky_coord = SkyCoord(ra=ra_str, dec=dec_str, unit=(u.hourangle, u.deg))
        ra_deg = sky_coord.ra.degree
        dec_deg = sky_coord.dec.degree
        return ra_deg, dec_deg
    except Exception as e:
        print(f"Error converting coordinates RA_DEC: {ra_dec_str}: {e}")
        return np.nan, np.nan

def convert_size(size_str):
    """Converts size string to decimal degrees."""
    try:
        if not size_str:
            return np.nan
        size_str = ''.join(filter(lambda x: x.isdigit() or x == '.', size_str))
        major_size = float(size_str.split('.')[0] + '.' + ''.join(size_str.split('.')[1:]))  # Handle multiple decimal points
        return major_size / 60.0  # Convert arcminutes to degrees
    except Exception as e:
        print(f"Error converting size: {size_str}: {e}")
        return np.nan

def main():
    # Read the existing input CSV
    input_data = pd.read_csv(input_csv)

    # Fetch the classical catalog
    classical_catalog = fetch_classical_catalog()

    # Combine the two dataframes
    combined_data = pd.concat([input_data, classical_catalog], ignore_index=True)

    # Write the combined data to a new CSV
    combined_data.to_csv(output_csv, index=False)

    print(f"Combined catalog saved to {output_csv}")

if __name__ == '__main__':
    main()
