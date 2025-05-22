'''
Model validation: compare the modelled surface currents with the Pelagia measurements
For this you need: 
- the model output (croco_avg.nc file for January 2024)
- Extracted Pelagia data (Pelagia_transects_data.npz, made with script 1_PE529_extract_surface_currents.py)
- shape of Curacao as shapefile for plotting (CUW_adm0.shp), found at www.gadm.org, contributor: OCHA Field Information Services Section (FISS), available publicly
'''

#%% Import libraries

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean as cmo
import cartopy.crs as ccrs
import pandas as pd
import cmocean
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

#%% Load Pelagia data and shape of Curacao

data          = np.load('data/Pelagia_transects_data.npz') # this data is created with another script: 1_PE529_extract_surface_currents.py
UVEL_all      = data['UVEL_all']
VVEL_all      = data['VVEL_all']
LATITUDE_all  = data['LATITUDE_all']
LONGITUDE_all = data['LONGITUDE_all']
TIME_all      = data['TIME_all']
speed_all     = data['speed_all']
dir_all       = data['dir_all']

# shapefile of Curacao
shapefile_path = 'data/CUW_adm0.shp'
land           = gpd.read_file(shapefile_path)

#%% Load and process SCARIBOS model data:

month        = 'Y2024M01'
dir          = '/nethome/berto006/croco/CONFIG/SCARIBOS_V8/CROCO_FILES/'
scarib_file  = f"{dir}croco_avg_{month}.nc"
ds_scarib    = xr.open_dataset(scarib_file)
u            = ds_scarib.u
v            = ds_scarib.v
mask_rho     = ds_scarib.mask_rho
lon_scarib   = ds_scarib.lon_u.values
lat_scarib   = ds_scarib.lat_u.values

# Include only the period of the Pelagia measurements
time_array     = ds_scarib.time.values
origin_time    = pd.Timestamp('2000-01-01')
timedeltas     = pd.to_timedelta(time_array, unit='s')
readable_times = origin_time + timedeltas
index_start    = np.where(readable_times == pd.Timestamp('2024-01-04 00:30:25'))[0][0] # find how many hours after was the 1st jan 2024
index_end      = np.where(readable_times == pd.Timestamp('2024-01-23 00:30:25'))[0][0] # find how many hours after was the 1st jan 2024
u_top          = u.isel(s_rho=-1).isel(time=slice(index_start, index_end))
v_top          = v.isel(s_rho=-1).isel(time=slice(index_start, index_end))

# calculate speed:
sq_u  = u_top**2
sq_v  = v_top**2
sq_u.coords["xi_u"]    = sq_u.coords["xi_u"] - 0.5
sq_u.coords["eta_rho"] = sq_u.coords["eta_rho"] + 0.5
sq_v.coords["xi_rho"]  = sq_v.coords["xi_rho"]
sq_v.coords["eta_v"]   = sq_v.coords["eta_v"]
sq_u                   = sq_u.rename({"xi_u": "xi_rho", "eta_rho": "eta_v"})
add_sq                 = sq_u + sq_v
add_sq_round           = add_sq.round(2)
speed_scarib           = np.sqrt(add_sq).mean("time")
u_top_mean             = u_top.mean("time")
v_top_mean             = v_top.mean("time")
speed                  = speed_scarib.drop(["lon_v", "lat_v"])
max_speed_scarib       = speed_scarib.max().values
print(f"Maximum speed of the model is: {speed_scarib.max().values} m/s")

# save calculated speed, u, v and mask_rho as netcdf (for quick plotting later if needed)
speed_scarib.to_netcdf(f"data/speed_scarib_avg_{month}.nc")
u_top_mean.to_netcdf(f"data/u_scarib_avg_{month}.nc")
v_top_mean.to_netcdf(f"data/v_scarib_avg_{month}.nc")
mask_rho.to_netcdf(f"data/mask_rho_scarib_{month}.nc")

# rename for plotting:
u_scarib = u_top_mean
v_scarib = v_top_mean

#%% Interpolate SCARIBOS data to Pelagia locations

# Load SCARIBOS grid data
dataset = ds_scarib
lon_grid = dataset.lon_u.values  # 2D array
lat_grid = dataset.lat_u.values  # 2D array
u_grid = u_scarib  # 2D array
v_grid = v_scarib  # 2D array

# Load PLEAGIA lon/lat pairs (replace with actual data)
pleagia_lon = LONGITUDE_all
pleagia_lat = LATITUDE_all

# Create interpolators
u_interp_func = RegularGridInterpolator((lat_grid[:, 0], lon_grid[0, :]), u_grid[:-1,:])
v_interp_func = RegularGridInterpolator((lat_grid[:, 0], lon_grid[0, :]), v_grid[:,:-1])

# Interpolate u and v at PLEAGIA locations
pleagia_points = np.vstack((pleagia_lat, pleagia_lon)).T
u_interp = u_interp_func(pleagia_points)
v_interp = v_interp_func(pleagia_points)

#%% Plotting

cmap          = cmocean.cm.speed
scale         = 8  # Same scale for both plots
width_qui     = 0.0035
headwidth_qui = 5
lat_min       = 11.76875
lat_max       = 12.74791667
lon_min = -69.58125
lon_max = -68.46041667
max_speed     = 1.31 # for colorbar
skip          = (slice(None, None, 12), slice(None, None, 12))
scale         = 8  # Same scale for both plots
width_qui     = 0.0035
headwidth_qui = 5
font0         = 12


# FIGURE
fig, axs = plt.subplots(1, 2, figsize=(11, 6), subplot_kw={'projection': ccrs.PlateCarree()})

cf = axs[0].contourf(lon_scarib, lat_scarib, speed_scarib, 200, 
                     cmap=cmap, vmin=0, vmax=max_speed)
for c in cf.collections:
    c.set_rasterized(True)
axs[0].set_title('(a) SCARIBOS model')
axs[0].set_xticks(np.round(np.arange(lon_scarib.min(), lon_scarib.max(), 0.5), 2))
axs[0].set_yticks(np.arange(lat_scarib.min(), lat_scarib.max(), 0.3))
axs[0].quiver(pleagia_lon[::4], pleagia_lat[::4], u_interp[::4], v_interp[::4], 
              color="black", scale=scale, width=width_qui, headwidth=headwidth_qui)
axs[0].add_geometries(land.geometry, crs=ccrs.PlateCarree(), facecolor='white', edgecolor='none', zorder=2)
axs[0].set_xlim([depth_lon_min, depth_lon_max])
axs[0].set_ylim([depth_lat_min, depth_lat_max])
axs[0].tick_params(axis='both', which='major', labelsize=font0)
axs[0].set_yticklabels(['{:.1f}° N'.format(x) for x in axs[0].get_yticks()])
axs[0].set_xticklabels(['{:.1f}° W'.format(abs(x)) for x in axs[0].get_xticks()])



axs[1].add_geometries(land.geometry, crs=ccrs.PlateCarree(), facecolor='grey', edgecolor='none', zorder=2)
axs[1].set_xticks(np.round(np.arange(lon_scarib.min(), lon_scarib.max(), 0.5), 2))
axs[1].tick_params(axis='both', which='major', labelsize=font0)

norm = mcolors.Normalize(vmin=0, vmax=max_speed)
quiver = axs[1].quiver(LONGITUDE_all[::4], LATITUDE_all[::4], UVEL_all[::4], VVEL_all[::4], speed_all[::4], 
                       scale=scale, cmap=cmo.cm.speed, width=width_qui+0.001, headwidth=headwidth_qui, norm=norm)
axs[1].set_title('(b) ADCP data')
axs[1].set_xlim([depth_lon_min, depth_lon_max])
axs[1].set_ylim([depth_lat_min, depth_lat_max])
axs[1].set_xticklabels(['{:.1f}° W'.format(abs(x)) for x in axs[1].get_xticks()])


cbar_ax = fig.add_axes([0.076, 0.12, 0.899, 0.035])  # [left, bottom, width, height]
cbar = fig.colorbar(quiver, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Speed [m/s]', fontsize=font0)
cbar.set_ticks([0,max_speed / 4,  max_speed / 2, max_speed *3/ 4, max_speed])
cbar.set_ticklabels([f"{0:.1f}", f"{max_speed / 4:.1f}", f"{max_speed / 2:.1f}", f"{max_speed * 3 / 4:.1f}", f"{max_speed:.1f}"])
cbar.ax.tick_params(labelsize=font0)

plt.tight_layout(rect=[0, 0.05, 1, 1.1])  # Leave space at the bottom for the colorbar

# save
plt.savefig('figures/validation_SCARIBOS_vs_Pelagia.png', dpi=300)



