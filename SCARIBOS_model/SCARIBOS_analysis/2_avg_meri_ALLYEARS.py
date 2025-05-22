'''
Script to process and plot meridional flow cross seciton analysis - average of all years
Author: vesnaber
Needed to run the script:
- SCARIBOS output files: croco_avg_{month}.nc
With this script you also save the average meridional flow for each month in a separate file: croco_avg_meri_{month}.nc 
--> used for plotting (so that the calculations do not have to be re-made)
'''

#%% 
# Imports
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import cmocean


# STEP 1: Calculate the mean meridional velocity profile for each month and store it in a separate .nc file

# configuration name of croco model
config = 'SCARIBOS_V8'

# List of simulation months
sim_months = ['Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
              'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
              'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
              'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
              'Y2024M01', 'Y2024M02', 'Y2024M03']

#%%
# Accumulate mean meridional velocity profiles for all months
all_means  = []
min_values = []
max_values = []

# Loop through each simulation month
for i, sim_month in enumerate(sim_months):
    # Load CROCO output file for the current month
    his_file = f'../CONFIG/{config}/CROCO_FILES/croco_avg_{sim_month}.nc'
    print(f'Opening file: {his_file}')
    ds_scarib = xr.open_dataset(his_file)
    print(f"Processing {sim_month}")

    # Calculate mean meridional velocity profile for the current month
    u = ds_scarib.u.isel(xi_u=150).isel(eta_rho=slice(145, 225))
    u_meri_mean = u.mean("time")

    # Append to the list of all monthly means
    all_means.append(u_meri_mean)

    # Save the mean of the current month to a separate NetCDF file
    u_meri_mean.to_netcdf(f'../../../croco/CONFIG/{config}/CROCO_FILES/croco_avg_meri_{sim_month}.nc')
    print(f'Mean for {sim_month} saved.')
    min_values.append(u_meri_mean.min().values)
    max_values.append(u_meri_mean.max().values)


min_values = np.array(min_values)
max_values = np.array(max_values)
min_all = min_values.min()
max_all = max_values.max()
print(f"Min value for all months: {min_all}")
print(f"Max value for all months: {max_all}")

# mean for all months combined and saved
u_meri_mean_all = xr.concat(all_means, dim="time").mean("time")
u_meri_mean_all.to_netcdf(f'../CONFIG/{config}/CROCO_FILES/croco_avg_meri_ALLYEARS.nc')
print("All months combined mean saved.")


#%%

# STEP 2: plot the average meridional velocity profile for all years together 

ds_scarib   = xr.open_dataset(f"../../../croco/CONFIG/{config}/CROCO_FILES/croco_avg_Y2020M04.nc") # open one croco file just so you can plot bathymetry cross section
u_meri_mean = xr.open_dataset(f"../../../croco/CONFIG/{config}/CROCO_FILES/croco_avg_meri_ALLYEARS.nc")

s_rho               = u_meri_mean.s_rho
h                   = ds_scarib.h
h_meri              = h.sel(xi_rho=150)
h_meri              = h_meri.sel(eta_rho=slice(145, 225))
depth               = h_meri * s_rho
depth_bottom        = depth.isel(s_rho=0)
depth_bottom_values = depth_bottom.values
minus_1400          = -1400 * np.ones_like(depth_bottom_values)
u_meri_mean.coords["depth"] = depth

# figure
fig = plt.figure(figsize=(28.5, 13.5))

image = u_meri_mean.u.plot(x="lat_u", y="depth",  add_colorbar=False, vmin=-0.8, vmax=0.8, cmap=cmocean.cm.balance, rasterized=True)
depth_bottom.plot(color='grey', x="lat_rho")
plt.title("(a) Average zonal velocity at 69\u00b0 W meridional cross-seciton, April 2020-March 2024", fontsize=26, pad=20)
plt.fill_between(depth.lat_rho, depth_bottom_values, minus_1400, color='gray', alpha=0.4)
plt.xlim([11.42, 12.149])
plt.ylim([-1400, 0])
plt.ylabel('Depth [m]', fontsize=22)
plt.xlabel('', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.gca().set_xticklabels(['{:.1f}° N'.format(abs(x)) for x in plt.gca().get_xticks()])


cbar = fig.colorbar(image)
cbar.set_label('Zonal velocity [m/s]', fontsize=22)
cbar.ax.tick_params(labelsize=20)

plt.savefig(f'{config}_avg_meri_ALLYEARS_HQ.png',bbox_inches="tight", dpi=300)
plt.savefig(f'{config}_avg_meri_ALLYEARS_HQ.pdf',bbox_inches="tight", dpi=300)




# %%
