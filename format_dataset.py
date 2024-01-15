import csv

def transform_csv(input_csv, output_csv):
    with open(input_csv, mode='r', encoding='utf-8') as infile, open(output_csv, mode='w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile, delimiter=';')
        fieldnames = ['Name', 'Gal. Coord.', 'RA', 'DEC', 'Size', 'Statut']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in reader:
            print("Processing row:", row)  # Debugging statement
            # Skip rows with 'Non NP' in the 'STATUT' column or invalid rows
            if 'Non NP' in row['STATUT'] or 'Galaxie' in row['STATUT'] or 'Région HII' in row['STATUT'] or '-------' in row['AD']:
                continue

            # Transform and write the row
            new_row = {
                'Name': row['NOM'],
                'Gal. Coord.': '',  # Placeholder as original data doesn't contain this
                'RA': row['AD'],  # Corrected column name for Right Ascension
                'DEC': row['DEC'],
                'Size': row['DIMENSION'],
                'Statut': row['STATUT']
            }
            writer.writerow(new_row)

# Replace with your actual file paths
transform_csv('planetary_nebula_all.csv', 'planetary_nebula_all_formated.csv')
