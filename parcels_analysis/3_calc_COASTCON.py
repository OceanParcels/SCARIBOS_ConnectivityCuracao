'''
Script to calculate connectivity between zones for the COASTCON scenario (= Intra-island connectivity)
This script runs in parallel, each zone is calculating in parallel and months are looped through. 
Needed files: 
- buffer areas of the zones (COASTCON_buffered_areas.npy), created with 1_buffer_areas_COASTCON.py
- zarr files with particle locations, created with 2_run_COASTCON_ZONE*.py
'''

import os
import sys
import numpy as np
import xarray as xr
from shapely.geometry import Point
import zarr


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <zone_num>")
        sys.exit(1)
    
    # Read zone number from command-line arguments
    zone_num = int(sys.argv[1])
    release_location = f'zone_{zone_num}'

    # load buffer areas:
    buffered_areas = np.load('COASTCON/buffer_areas/COASTCON_buffered_areas.npy', allow_pickle=True)

    # List of year-month combinations
    years_months_list = ['Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
                        'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
                        'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
                        'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
                        'Y2024M01', 'Y2024M02']

    # Initialize dictionary to store results for all zones and destinations
    results = {i: [] for i in range(len(buffered_areas))}

    # Loop over months in the years_months_list
    for month in years_months_list:
        file_path = f'OUT_COASTCON/COASTCON_{release_location}_{month}.zarr'

        if not os.path.exists(file_path):
            print(f"Zarr file not found: {file_path}")
            continue  # Skip if file doesn't exist

        try:
            ds = xr.open_zarr(file_path)
            part_num_end = ds.trajectory.size
            print(f'Processing {month} with {part_num_end} particles...')

            lon_values = ds.lon[:, :].values
            lat_values = ds.lat[:, :].values
            print('Loaded lon and lat values')

            # Create shapely points
            points = [Point(lon, lat) for lon, lat in zip(lon_values.ravel(), lat_values.ravel())]

            # Loop over buffered areas and calculate intersections
            for i, buffer_area in enumerate(buffered_areas):
                print(f"Calculating intersections for zone {i}...")

                # Check intersection for each point
                intersects = np.array([point.intersects(buffer_area) for point in points]).reshape(lon_values.shape)
                particles_entered_area = np.any(intersects, axis=1)
                portion_entering = np.mean(particles_entered_area)

                # Store the portion entering for this zone (i) and month
                results[i].append((month, portion_entering))
                print(f"Month {month}, Destination zone: {i}: {portion_entering * 100:.2f}% particles entered the area")

        except Exception as e:
            print(f"Failed to process {month} due to: {e}")

    # After looping through all months, save results to a single file for each destination
    for i in range(len(buffered_areas)):
        np.save(f'COASTCON/connectivity/COASTCON_source_{release_location}_destination_zone_{i}.npy', results[i])

    print(f'All months processed successfully for zone {zone_num}!')

