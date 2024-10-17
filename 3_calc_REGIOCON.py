'''
Script to calculate the portion of particles entering the buffered area surrounding CUracao
for each month of a given year and release location.
Author: vesnaber
Files needed:
- coastal_points_CURACAO.npy: numpy file with the coastal points of Curacao
- OUT_REGIOCON/REGIOCON_{release_location}_{month}.zarr: zarr files with 
    the particle trajectories, created with 2_run_REGIOCON_{release_location}.py
Calculated connectivity is saved in a numpy file for each release location 
and year (all months in a given year are saved in this year).
'''

# import
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import sys
import zarr

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <location> <year>; if you see this message, you forgot to add the location and year as arguments!")
        sys.exit(1)

    location = sys.argv[1]
    year = sys.argv[2]

    # Define release locations based on the location argument
    if location == '1':
        release_location = 'BONA'
    elif location == '2':
        release_location = 'ARUB'
    elif location == '3':
        release_location = 'VEIS'
    else:
        release_location = 'VECO'

    print(f"Release location: {release_location}")

    # Define months based on the year argument
    months_dict = {
        '2020': ['Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12'],
        '2021': ['Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12'],
        '2022': ['Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12'],
        '2023': ['Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12'],
        '2024': ['Y2024M01', 'Y2024M02']
    }

    months = months_dict.get(year, [])
    if not months:
        print(f"Invalid year: {year}")
        sys.exit(1)

    # Load coastal points and buffer area
    curacao_coast_points = np.load('INPUT/coastal_points_CURACAO.npy')
    points = [Point(lon, lat) for lon, lat in zip(curacao_coast_points[0], curacao_coast_points[1])]
    multi_point = MultiPoint(points)
    buffer_distance = 0.05  # Define the buffer distance
    buffered_area = multi_point.buffer(buffer_distance)
    buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_area])
    print('Buffered area created!')

    portion_entering = []

    for month in months:
        print(f"Processing {month} for {release_location}...")

        file_path = f'/nethome/berto006/surface_run_parcels/OUT_REGIOCON/REGIOCON_{release_location}_{month}.zarr'
        
        print(f"File path: {file_path}")

        # Try to open the file with xarray, fallback to zarr if there's an error
        try:
            ds = xr.open_zarr(file_path, drop_variables=['particle_age', 'obs', 'time', 'z', 'trajectory'])
        except ValueError as e:
            print(f"Encountered ValueError with xr.open_zarr: {e}")
            print("Switching to zarr directly...")
            ds = zarr.open(file_path, mode='r')
            lon_values = ds['lon'][:, :]
            lat_values = ds['lat'][:, :]
        else:
            print('Using xarray...')
            lon_values = ds.lon[:, :].values
            lat_values = ds.lat[:, :].values

        print('Lon and lat values loaded!')

        # Create points and calculate intersections with buffered area
        points = [Point(lon, lat) for lon, lat in zip(lon_values.ravel(), lat_values.ravel())]
        intersects = np.array([point.intersects(buffered_area) for point in points])
        intersects = intersects.reshape(lon_values.shape)
        print('Intersections calculated!')

        # Calculate the portion of particles entering the buffered area
        particles_entered_area = np.any(intersects, axis=1)
        portion_entering.append(np.mean(particles_entered_area))
        print(f"Portion of particles entering the area: {np.mean(particles_entered_area)}")
    
    # save portion_entering to file once all months are processed
    np.save(f'/nethome/berto006/surface_run_parcels/REGIOCON/connectivity/REGIOCON_source_{release_location}_{year}.npy', portion_entering)
    print(f"Portion entering saved for {release_location} in {year}!")

    print('All months processed!')
