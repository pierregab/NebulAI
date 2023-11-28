import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from matplotlib.path import Path

def point_in_hull(point, hull):
    path = Path(hull)
    return path.contains_points([point])

def read_data(file_path):
    df = pd.read_csv(file_path)
    return SkyCoord(df['RA'], df['DEC'], unit=(u.hourangle, u.deg))

def plot_sky(coords, title, NorthenHemisphere=False, plot = True):
    norm_index = np.arange(len(coords)) / (len(coords) - 1)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(111, projection='mollweide')
    scat = ax.scatter(coords.ra.wrap_at(180*u.deg).radian, coords.dec.radian, c=norm_index, cmap=cmap)
    ax.grid(False)

    points = np.array([coords.ra.wrap_at(180*u.deg).radian, coords.dec.radian]).T
    hull = ConvexHull(points, qhull_options="QbB Qt Q12")
    min_ra, max_ra = np.min(points[hull.vertices, 0]), np.max(points[hull.vertices, 0])
    min_dec, max_dec = np.min(points[hull.vertices, 1]), np.max(points[hull.vertices, 1])
    ra_grid = np.arange(min_ra, max_ra, 5 * u.deg.to(u.rad))

    if NorthenHemisphere == True:
        dec_grid_start = 0
    else:
        dec_grid_start = min_dec

    dec_grid = np.arange(dec_grid_start, max_dec, 5 * u.deg.to(u.rad))

    gridline_squares = []
    for ra_start in ra_grid:
        ra_end = ra_start + 5 * u.deg.to(u.rad)
        for dec_start in dec_grid:
            dec_end = dec_start + 5 * u.deg.to(u.rad)
            if point_in_hull([ra_start, dec_start], points[hull.vertices]):
                ra_min, ra_max = max(ra_start, min_ra), min(ra_end, max_ra)
                dec_min, dec_max = max(dec_start, min_dec), min(dec_end, max_dec)
                gridline_squares.append((ra_min, ra_max, dec_min, dec_max))
                ax.plot([ra_min, ra_min], [dec_min, dec_max], color='gray', linestyle='-', linewidth=0.5)
                ax.plot([ra_min, ra_max], [dec_min, dec_min], color='gray', linestyle='-', linewidth=0.5)
                ax.plot([ra_max, ra_max], [dec_min, dec_max], color='gray', linestyle='-', linewidth=0.5)
                ax.plot([ra_min, ra_max], [dec_max, dec_max], color='gray', linestyle='-', linewidth=0.5)

    if plot == True:
        hull_poly = plt.Polygon(points[hull.vertices], alpha=0.2, edgecolor='black', facecolor='blue')
        ax.add_patch(hull_poly)
        cb = plt.colorbar(scat)
        cb.set_label('Newest discovery', fontsize=10)
        ax.set_xlabel('Right Ascension (degrees)', fontsize=12)
        ax.set_ylabel('Declination (degrees)', fontsize=12)
        plt.title(title, fontsize=14, y=1.08)
        plt.show()

    if plot == False:
        # del the fig 
        plt.close(fig)

    return np.array(gridline_squares) * u.rad.to(u.deg)

def main():
    coords = read_data('output33.csv')
    gridline_squares_full = plot_sky(coords, 'Sky Map of Galactic Planetary Nebula Candidates and search zone gridlines', plot = False)
    gridline_squares_north = plot_sky(coords, 'Sky Map of Northern Galactic Planetary Nebula Candidates and search zone gridlines', NorthenHemisphere=True)

if __name__ == "__main__":
    main()
