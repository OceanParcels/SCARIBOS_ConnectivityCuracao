'''
Model validation, for Supplimentary Figure Fig. S4
SST comparison between SCARIBOS and GLORYS for 2022
Download before running the script: GLORYS dataset
'''


#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean as cmo
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec

dir   = 'croco/CONFIG/SCARIBOS_V8/CROCO_FILES/surface_currents/'
months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
ds_scaribos_sss = xr.concat([xr.open_dataset(f'croco/CONFIG/SCARIBOS_V8/CROCO_FILES/surface_currents/Y2022M{month}_sst.nc') for month in months], dim='time')
ds_scaribos_sss = ds_scaribos_sss.where(ds_scaribos_sss.temp != 0)

# Load all the monthly-mean NetCDF files
globcur_file_pattern = 'croco/ANALYSIS/VALIDATION/data/GLORYS_MONTHLY/*.nc'
ds_coper = xr.open_mfdataset(globcur_file_pattern, combine='by_coords')

lon_slice = slice(-70.5, -66)
lat_slice = slice(10, 13.5)

sss_all_months = []
for t in range(len(ds_coper.time-1)):
    sss = ds_coper.thetao.isel(time=t).sel(longitude=lon_slice, latitude=lat_slice)
    sss = sss.where(sss != 0)
    sss_all_months.append(sss)

sss_all_months = xr.concat(sss_all_months, dim='time')

# %% regrid SST of SCARIBOS

lon_regg = sss_all_months.longitude.values 
lat_regg = sss_all_months.latitude.values 
lon_regg2d, lat_regg2d = np.meshgrid(lon_regg, lat_regg)
points_target = np.column_stack((lon_regg2d.ravel(), lat_regg2d.ravel()))

sss_regg = []

for t in range(len(ds_scaribos_sss.time)):
    sss = ds_scaribos_sss.temp.isel(time=t)

    # Extract original 2D lon/lat grids
    lon_u_2d = ds_scaribos_sss.lon_rho.values
    lat_u_2d = ds_scaribos_sss.lat_rho.values

    # Flatten original grid and values
    points_source = np.column_stack((lon_u_2d.ravel(), lat_u_2d.ravel()))
    values = sss.values.ravel()

    # Perform interpolation
    sss_interp = griddata(points_source, values, points_target, method="linear")

    # Reshape back to 2D
    sss_interp_2d = sss_interp.reshape(lon_regg2d.shape)

    # Store as xarray DataArray
    sss_regg.append(xr.DataArray(sss_interp_2d, dims=("lat", "lon"), coords={"lat": lat_regg, "lon": lon_regg}))

sss_regg = xr.concat(sss_regg, dim="time")
sss_regg = sss_regg.assign_coords(time=ds_scaribos_sss.time)
sss_regg = sss_regg.where(sss_regg != 0)

#%% FIGURE

vmin, vmax = 23, 31
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


fig = plt.figure(figsize=(11.5, 14.4), constrained_layout=True)
gs = gridspec.GridSpec(6, 5, width_ratios=[1, 1, 1,1, 0.1], wspace=0.1)
axs = np.array([[fig.add_subplot(gs[row, col]) for col in range(4)] for row in range(6)])

for row in range(6): 
    for col in [0, 1, 2, 3]:  
        axs[row, col].set_aspect('equal')
        month_index = row * 2 + (col // 2) 
        if month_index >= 12:
            continue

        if col % 2 == 0:  # SCARIBOS
            sss = sss_regg.isel(time=month_index)
            pcm = axs[row, col].pcolormesh(
                sss.lon, sss.lat, sss,
                cmap=cmo.cm.thermal, vmin=vmin, vmax=vmax, rasterized=True
            )
            axs[row, col].text(-70.3, 10.1, "SCARIBOS", fontsize=11, ha='left', va='bottom')

            axs[row, col].set_xlim(-70.5, -66)
            axs[row, col].set_ylim(10, 13.5)

        else:  # GlobCurrent
            sss = sss_all_months.isel(time=month_index)
            pcm = axs[row, col].pcolormesh(
                sss.longitude, sss.latitude, sss[0,:,:],
                cmap=cmo.cm.thermal, vmin=vmin, vmax=vmax, rasterized=True
            )
            axs[row, col].text(-70.3, 10.1, "GLORYS12V1", fontsize=11, ha='left', va='bottom')

            axs[row, col].set_xlim(-70.5, -66)
            axs[row, col].set_ylim(10, 13.5)

        axs[row, col].set_aspect('equal')

    # Add month and year annotations
    axs[row, 0].text(-65.85, 13.6, months[row * 2], fontsize=12, ha='right', va='bottom', fontweight='bold')
    axs[row, 1].text(-70.6, 13.6, "2022", fontsize=12, ha='left', va='bottom', fontweight='bold')
    axs[row, 2].text(-65.85, 13.6, months[row * 2 + 1], fontsize=12, ha='right', va='bottom', fontweight='bold')
    axs[row, 3].text(-70.6, 13.6, "2022", fontsize=12, ha='left', va='bottom', fontweight='bold')

# Formatting ticks and labels
for row in range(6):
    for col in range(4):
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

    axs[row, 0].set_yticks(np.arange(10, 14, 1))
    axs[row, 0].set_yticklabels([f"{tick:.0f}° N" for tick in np.arange(10, 14, 1)], fontsize=11)
    axs[row, 0].set_aspect('equal')

    if row == 5:  # Add x-ticks for the last row
        for col in range(4):
            axs[row, col].set_xticks(np.arange(-70, -65, 1))
            axs[row, col].set_xticklabels([f"{abs(tick):.0f}° W" for tick in np.arange(-70, -65, 1)], rotation=90, fontsize=11)
            axs[row, col].set_aspect('equal')

cbar_ax = fig.add_subplot(gs[:, 4])  
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='vertical')
cbar.set_label("Sea surface temperature [°C]", fontsize=11)
cbar.ax.tick_params(labelsize=11)

# save
plt.savefig('figures/SCARIBOS_VS_GLORYS_SST_regridded.png', dpi=300, bbox_inches='tight')
