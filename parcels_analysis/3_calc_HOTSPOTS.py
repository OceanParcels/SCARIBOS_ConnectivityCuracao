'''
Script to calculate probability density functions (PDF) for the HOTSPOTS scenario
and storing the results as NetCDF files and figures (seperate figures of each month)
parcels scenario name: HOTSPOTS
'''

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cmocean
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

# load
path             = "OUT_HOTSPOTS/"          # parcels output is here
probability_path = "HOTSPOTS/PDF_unfiltered/"    # calculated PDFs are here
probability_plots_path = "HOTSPOTS/PDF_unfiltered_figures/"
config           = 'SCARIBOS_V8'             # model configuration (croco)
path_grd         = '~/croco/CONFIG/'+config+'/CROCO_FILES/' # path of grid file
part_config      = 'HOTSPOTS'               # name of scenario
xmin, xmax = -69.49, -68.49                 # region of interest - in this region the PDF is caculated
ymin, ymax = 11.67, 12.67                   # --|--
bins_x, bins_y = 600, 500                   # resolution of PDF

# List of simulation months
sim_months_list = [
    'Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
    'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
    'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
    'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
    'Y2024M01', 'Y2024M02'
]

def probability_density(ds, bins_x, bins_y):
    """
    Calculates the PDF of the longitude and latitude coordinates of the trajectories at each observation.
    
    Input variables:
    - ds: OceanParcels output of lon & lat of each particle at each timestep
    - bins_x & bins_y: number of bins in x/y direction 

    Function based on work by: Jimena Medina Rubio
    """
    
    def histogram(lon, lat, bins_x, bins_y): 
        # Filter out NaN values
        lon = lon[~np.isnan(lon)]
        lat = lat[~np.isnan(lat)]
        # Define the edges of the bins
        bins_edges_x = np.histogram_bin_edges(lon, bins=bins_x)
        bins_edges_y = np.histogram_bin_edges(lat, bins=bins_y)
        # Calculate the 2D normalized histogram & bin edges
        H, x, y = np.histogram2d(lon.flatten(), lat.flatten(), bins=[bins_edges_x, bins_edges_y], density=True)
        return H, x, y

    # Apply histogram function to all trajectories at every observation
    lon_filter = ds.lon.values
    lat_filter = ds.lat.values

    result = xr.apply_ufunc(
        histogram,
        lon_filter,
        lat_filter,
        bins_x,
        bins_y,
        input_core_dims=[['traj', 'obs'], ['traj', 'obs'], [], []],
        output_core_dims=[['binx', 'biny'], [], []],
        dask='parallelized',
        vectorize=True,
        output_dtypes=[float])
    
    # Define the bin centers from the output bin edges
    bins_centres_x = np.linspace(result[1][0], result[1][-1], len(result[1])-1)
    bins_centres_y = np.linspace(result[2][0], result[2][-1], len(result[2])-1)
    # Convert particle counts per grid cell into a DataArray
    da_result = xr.DataArray(result[0], 
                             dims=['binx', 'biny'], 
                             coords={'binx': bins_centres_x, 'biny': bins_centres_y}, 
                             name='probability') 
    
    # Set values equal to zero to NaN & normalize results so that the sum of probability = 100
    da_result = da_result.where(da_result != 0, np.nan) * 100 / np.nansum(da_result)
    probability_file = f'{probability_path}{part_config}_PDF_unfiltered_x600_y500_{sim_months}.nc'
    # Save the DataArray to NetCDF
    da_result.to_netcdf(probability_file)
    return da_result.T


# Loop over all months and calculate PDF
for sim_months in sim_months_list:
    print(f"Processing {sim_months}...")
    
    # Load the data for the current month
    file = (path + part_config + "_" + sim_months + ".zarr")
    ds = xr.open_zarr(file)
    
    # Calculate the PDF
    pdf = probability_density(ds, bins_x, bins_y)
    print(f"Calculated and saved probability data for {sim_months} as NetCDF")

    # Plot the PDF
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cmocean.cm.matter
    pcm = pdf.plot(ax=ax, cmap=cmap, norm=LogNorm(vmin=5e-4, vmax=5e-3), add_colorbar=False)
    cbar = fig.colorbar(pcm, ax=ax, extend='both')
    cbar.set_label('Probability Density')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    title = f'Probability Density Function of {part_config} particles for {sim_months}'
    ax.set_title(title)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

    plt.savefig(probability_plots_path + 'PDF_unfiltered_x600_y500_' + part_config + '_' + sim_months + '.png', dpi=300)
    plt.close(fig) 
    print(f"Saved figure for {sim_months}")