'''
Script to prepare consecutive files that will be used to run Parcels with CROCO
This version processes croco_avg.nc files for the years 2020-2023 and selects only the top layer (2D).
Author: vesnaber
Date of creation: 2024-04-26
'''

# Import necessary libraries
import numpy as np
from glob import glob
import xarray as xr
import os

print('Starting the job... :) ')

# Step 1: Loop over the years 2020 to 2023
years = range(2020, 2024)  # Years from 2020 to 2023
dir = '/nethome/berto006/croco/CONFIG/SCARIBOS_V8/CROCO_FILES/'

for year in years:
    # Step 2: Open files for each year
    files = sorted(glob(f'{dir}croco_avg_Y{year}M*.nc'))
    print(f"Processing files for the year {year}: {files}")

    # Step 3: Process each file and cut to the surface layer (top layer)
    for file in files:
        ds = xr.open_dataset(file)
        
        ds_surface  = ds.isel(s_rho=-1)
        month       = file.split('M')[1][:2]
        output_file = f"{dir}croco_avg_Y{year}M{month}_surface.nc"
        
        ds_surface.to_netcdf(output_file)
        print(f"Saved surface layer to {output_file}")
        
        ds.close()

print('Job completed successfully for all years! :)')

