'''
Script to process and plot surface currents - average of all years
Author: vesnaber
Needed to run the script:
- SCARIBOS output files: croco_avg_{month}.nc
With this script you also save the average surface flow for each month in a separate file
--> used for plotting (so that the calculations do not have to be re-made)
NOTE: If step one is already made, you can skip it and just run the plotting part (PART2, below).
'''

#%%

import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import numpy as np
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter

#%%
# STEP 1: Calculate the mean surface speed and velocities for each month and store it in a separate .nc file

# configuration name of croco model
config = 'SCARIBOS_V8'

sim_months = ['Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
              'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
              'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
              'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
              'Y2024M01', 'Y2024M02', 'Y2024M03']
#%%
# Accumulate mean surface speed and velocities for all months
u_mean_all  = []
v_mean_all  = []
speed_all   = []

for i, sim_month in enumerate(sim_months):
    year = int(sim_month[1:5])       # Extract year from sim_month
    month_idx = int(sim_month[6:8])  # Extract month index

    # Define output directory for the current month
    output_dir = f'../../../croco/CONFIG/{config}/CROCO_FILES/surface_currents/{sim_month}'
    u_file = f'{output_dir}_u.nc'
    v_file = f'{output_dir}_v.nc'
    speed_file = f'{output_dir}_speed.nc'

    # Load CROCO output file for the current month if not already processed
    his_file = f'../CONFIG/{config}/CROCO_FILES/surface_currents/croco_avg_{sim_month}_surface.nc'
    ds_scarib = xr.open_dataset(his_file)
    print(f"Processing {sim_month}")

    # Load and process data for the current month
    u = ds_scarib.u
    v = ds_scarib.v

    # Calculate squared velocity components
    u_top = u
    v_top = v
    sq_u = u_top**2
    sq_v = v_top**2

    # Adjust coordinates for grid alignment
    sq_u.coords["xi_u"] = sq_u.coords["xi_u"] - 0.5
    sq_u.coords["eta_rho"] = sq_u.coords["eta_rho"] + 0.5
    sq_v.coords["xi_rho"] = sq_v.coords["xi_rho"]
    sq_v.coords["eta_v"] = sq_v.coords["eta_v"]
    sq_u = sq_u.rename({"xi_u": "xi_rho", "eta_rho": "eta_v"})

    # Calculate the total squared speed and mean speed over time
    add_sq = sq_u + sq_v
    speed = np.sqrt(add_sq).mean("time")

    # Calculate mean u and v components over time
    u_top_mean = u_top.mean("time")
    v_top_mean = v_top.mean("time")

    # Drop unnecessary dimensions from speed
    speed = speed.drop(["lon_v", "lat_v"])

    # Append the mean u, v, and speed to the lists
    u_mean_all.append(u_top_mean)
    v_mean_all.append(v_top_mean)
    speed_all.append(speed)

    # Save u, v, and speed for the current month
    u_top_mean.to_netcdf(u_file)
    v_top_mean.to_netcdf(v_file)
    speed.to_netcdf(speed_file)
    print(f"Saved u, v, and speed for {sim_month}")

# Concatenate all monthly means
u_mean = xr.concat(u_mean_all, dim="time")
v_mean = xr.concat(v_mean_all, dim="time")
speed_all  = xr.concat(speed_all, dim="time")

# Calculate the mean of all monthly means
u_mean = u_mean.mean("time")
v_mean = v_mean.mean("time")
speed_all  = speed_all.mean("time")

# Save the mean of all monthly means to a separate NetCDF file
mean_file  = f'../CONFIG/{config}/CROCO_FILES/surface_currents/surface_currents_mean_of_all_years.nc'
speed_file = f'../CONFIG/{config}/CROCO_FILES/surface_currents/surface_speed_mean_of_all_years.nc'
u_mean.to_netcdf(mean_file)
v_mean.to_netcdf(mean_file)
speed_all.to_netcdf(speed_file)
print("All months combined mean saved.")


#%%
# STEP 2: plot the average surface currents for all years together

# Load the previously saved data
mean_file  = f'../../../croco/CONFIG/{config}/CROCO_FILES/surface_currents/surface_currents_mean_of_all_years.nc'
speed_file = f'../../../croco/CONFIG/{config}/CROCO_FILES/surface_currents/surface_speed_mean_of_all_years.nc'

# Load the datasets
ds_mean  = xr.open_dataset(mean_file)
ds_speed = xr.open_dataset(speed_file)

# Extract necessary variables
u_top_mean_all = ds_mean['u_mean']
v_top_mean_all = ds_mean['v_mean']
speed_masked_all = ds_speed['__xarray_dataarray_variable__']  # or use the correct name for the speed variable

skip = (slice(None, None, 16), slice(None, None, 16))
lon_grid = u_top_mean_all.lon_u.values
lat_grid = u_top_mean_all.lat_u.values

fig = plt.figure(figsize=(28.5, 13.5))
ax = plt.axes(projection=ccrs.PlateCarree())

# Plot speed
cmap = cmocean.cm.speed
image = speed_masked_all.plot(x="lon_u", y="lat_u", cmap=cmap, ax=ax, transform=ccrs.PlateCarree(),
                              cbar_kwargs={'label': 'Speed [m/s]', 'extend': 'max', 'extendrect': True}, vmin=0, vmax=1.33,  rasterized=True)

# Customize colorbar
cbar = image.colorbar
cbar.ax.tick_params(labelsize=20)
cbar.set_label('Speed [m/s]', fontsize=22)

# Mask u and v where both are zero
mask_u = (u_top_mean_all == 0) 
mask_v = (v_top_mean_all == 0)
u_quiv = u_top_mean_all.where(~mask_u)
v_quiv = v_top_mean_all.where(~mask_v)

# Add quiver plot for flow direction, only for non-zero u and v
quiv = ax.quiver(lon_grid[skip], lat_grid[skip], u_quiv[skip], v_quiv[skip],
                 transform=ccrs.PlateCarree(), color="black", scale=15, width=0.0015)

xl = ax.gridlines(draw_labels=True, linestyle='--', zorder=1)
xl.top_labels = False
xl.right_labels = False
xl.xlabel_style = {'size': 20}
xl.ylabel_style = {'size': 20}

# Add rectangular and line annotations
ax.plot([-69.5, -68.5, -68.5, -69.5, -69.5], [11.65, 11.65, 12.65, 12.65, 11.65], color='black', linewidth=3.5, transform=ccrs.PlateCarree(), zorder=3)
ax.plot([-69.0, -69.0], [11.45, 12.15], color='black', linewidth=3.5, transform=ccrs.PlateCarree(), zorder=3)

ax.set_ylim([10.35, None])
ax.set_title("(a) Average current speed and direction, April 2020-March 2024", fontsize=24, pad=20)


# Save the plot
plt.savefig(f"../../figures/{config}_avg_surface_ALLYEARS.png", bbox_inches="tight", dpi=300)
plt.savefig(f"../../figures/{config}_avg_surface_ALLYEARS.pdf", bbox_inches="tight", dpi=300)

print(f"Figure saved to figures/{config}_avg_surface_ALLYEARS...")


# %%
