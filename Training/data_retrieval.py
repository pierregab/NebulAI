import csv
import requests
from bs4 import BeautifulSoup

def scrape_data(url, output_filename):
    """
    Scrapes data from the given URL and writes it to a CSV file.
    
    :param url: The URL to scrape data from.
    :param output_filename: The name of the file to write the data to.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    td = soup.find_all('td', {'colspan': '3'})[5]
    table = td.find_parent('table')
    rows = table.find_all('tr')[12:]

    with open(output_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        headers = ['Name', 'Gal. Coord.', 'RA', 'DEC', 'Size', 'Statut']
        writer.writerow(headers)
        for row in rows:
            cells = row.find_all('td')
            writer.writerow([cell.get_text(strip=True) for cell in cells])

def retrieve_dataset():
    """
    Retrieves datasets by scraping specific URLs and writing the data to CSV files.
    """
    urls_and_filenames = [
        ("http://planetarynebulae.net/EN/page_np_resultat.php?id=372", "StDr.csv"),
        ("http://planetarynebulae.net/EN/page_np_resultat.php?id=167", "Ou.csv")
    ]
    for url, filename in urls_and_filenames:
        scrape_data(url, filename)

if __name__ == '__main__':
    retrieve_dataset()
