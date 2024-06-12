from astroquery.vo_conesearch import conesearch as vo_conesearch

def main():
    # Define the center coordinates (RA, Dec) and the search radius
    center = (324.0, 58.0)
    radius = 0.1  # Radius in degrees
    
    # Define the catalog database URL for IPHAS
    catalog_db = "http://vizier.u-strasbg.fr/viz-bin/votable/-A?-source=IPHAS2&-out.all&"
    
    try:
        # Perform the cone search
        search = vo_conesearch.conesearch(
            center=center,
            radius=radius,
            verb=3,
            catalog_db=catalog_db,
            return_astropy_table=True
        )
        
        # The result is already an Astropy Table
        result_table = search
        
        # Print the first few rows of the result table
        print(result_table)
        
    except Exception as e:
        print(f"Error performing cone search: {e}")

if __name__ == '__main__':
    main()
