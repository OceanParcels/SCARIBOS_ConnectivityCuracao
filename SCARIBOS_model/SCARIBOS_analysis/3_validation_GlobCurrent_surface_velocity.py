'''
Model validation, script to re-create Figure Fig. 2
Surface velocity comparison between SCARIBOS and GlobCurrent for 2022
Download before running the script: GLobCurrent dataset
'''

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import griddata
import cmocean as cmo
import matplotlib.gridspec as gridspec

# Load all the NetCDF files in the specified folder
globcur_file_pattern = 'data/GLOBCUR_2022/*.nc' # (download it to your own selected folder)
ds_globcur = xr.open_mfdataset(globcur_file_pattern, combine='by_coords')

# Subset coordinates
lon_slice = slice(-70.5, -66)
lat_slice = slice(10, 13.5)

# Calculate speed for all available months
speed_all_months = []
for t in range(len(ds_globcur.time)):
    uo = ds_globcur.uo.isel(time=t, depth=0).sel(longitude=lon_slice, latitude=lat_slice)
    vo = ds_globcur.vo.isel(time=t, depth=0).sel(longitude=lon_slice, latitude=lat_slice)
    speed = np.sqrt(uo**2 + vo**2)
    speed_all_months.append(speed)

# Combine all speeds into a single xarray.DataArray
speed_all_months = xr.concat(speed_all_months, dim='time')

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
ds_scaribos_u = xr.concat([xr.open_dataset(f'croco/CONFIG/SCARIBOS_V8/CROCO_FILES/surface_currents/Y2022M{month}_u.nc') for month in months], dim='time')
ds_scaribos_v = xr.concat([xr.open_dataset(f'croco/CONFIG/SCARIBOS_V8/CROCO_FILES/surface_currents/Y2022M{month}_v.nc') for month in months], dim='time')
ds_scaribos_speed = xr.concat([xr.open_dataset(f'croco/CONFIG/SCARIBOS_V8/CROCO_FILES/surface_currents/Y2022M{month}_speed.nc') for month in months], dim='time')


#%%
# regridding SCARIBOS: 

# Define target grid
lon_regg = ds_globcur.longitude.values  # 1D array
lat_regg = ds_globcur.latitude.values  # 1D array
lon_regg2d, lat_regg2d = np.meshgrid(lon_regg, lat_regg)  # 2D grid

# Flatten target grid for interpolation
points_target = np.column_stack((lon_regg2d.ravel(), lat_regg2d.ravel()))

# Regrid u
u_regg = []

for t in range(len(ds_scaribos_u.time)):
    u = ds_scaribos_u.u.isel(time=t)

    # Extract original 2D lon/lat grids
    lon_u_2d = ds_scaribos_u.lon_u.values
    lat_u_2d = ds_scaribos_u.lat_u.values

    # Flatten original grid and values
    points_source = np.column_stack((lon_u_2d.ravel(), lat_u_2d.ravel()))
    values = u.values.ravel()

    # Perform interpolation
    u_interp = griddata(points_source, values, points_target, method="linear")

    # Reshape back to 2D
    u_interp_2d = u_interp.reshape(lon_regg2d.shape)

    # Store as xarray DataArray
    u_regg.append(xr.DataArray(u_interp_2d, dims=("lat", "lon"), coords={"lat": lat_regg, "lon": lon_regg}))

# Convert list to xarray DataArray along time
u_regg = xr.concat(u_regg, dim="time")
u_regg = u_regg.assign_coords(time=ds_scaribos_u.time)

# same for v and speed
v_regg = []

for t in range(len(ds_scaribos_v.time)):
    v = ds_scaribos_v.v.isel(time=t)

    # Extract original 2D lon/lat grids
    lon_v_2d = ds_scaribos_v.lon_v.values
    lat_v_2d = ds_scaribos_v.lat_v.values

    # Flatten original grid and values
    points_source = np.column_stack((lon_v_2d.ravel(), lat_v_2d.ravel()))
    values = v.values.ravel()

    # Perform interpolation
    v_interp = griddata(points_source, values, points_target, method="linear")

    # Reshape back to 2D
    v_interp_2d = v_interp.reshape(lon_regg2d.shape)

    # Store as xarray DataArray
    v_regg.append(xr.DataArray(v_interp_2d, dims=("lat", "lon"), coords={"lat": lat_regg, "lon": lon_regg}))

# Convert list to xarray DataArray along time
v_regg = xr.concat(v_regg, dim="time")
v_regg = v_regg.assign_coords(time=ds_scaribos_v.time)

# same for speed
speed_regg = []

for t in range(len(ds_scaribos_speed.time)):
    speed = ds_scaribos_speed.__xarray_dataarray_variable__.isel(time=t)

    # Extract original 2D lon/lat grids
    lon_speed_2d = ds_scaribos_speed.lon_u.values
    lat_speed_2d = ds_scaribos_speed.lat_u.values

    # Flatten original grid and values
    points_source = np.column_stack((lon_speed_2d.ravel(), lat_speed_2d.ravel()))
    values = speed.values.ravel()

    # Perform interpolation
    speed_interp = griddata(points_source, values, points_target, method="linear")

    # Reshape back to 2D
    speed_interp_2d = speed_interp.reshape(lon_regg2d.shape)

    # Store as xarray DataArray
    speed_regg.append(xr.DataArray(speed_interp_2d, dims=("lat", "lon"), coords={"lat": lat_regg, "lon": lon_regg}))

# Convert list to xarray DataArray along time
speed_regg = xr.concat(speed_regg, dim="time")
speed_regg = speed_regg.assign_coords(time=ds_scaribos_speed.time)


#%% FIGURE

# if values zero --> NaN
speed_regg = speed_regg.where(speed_regg > 0)
u_regg = u_regg.where(u_regg != 0)
v_regg = v_regg.where(v_regg != 0)
vmin, vmax = 0, 1.3
quiver_skip = 20
quiver_scale = 5       # Consistent scale for both SCARIBOS and GlobCurrent quivers
quiver_width = 0.0025  # Slightly thicker quivers
months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Figure
fig = plt.figure(figsize=(11.5, 14.4), constrained_layout=True)
gs = gridspec.GridSpec(6, 5, width_ratios=[1, 1, 1,1, 0.1], wspace=0.1)

# Create axes for the plots
axs = np.array([[fig.add_subplot(gs[row, col]) for col in range(4)] for row in range(6)])

for row in range(6):
    for col in [0, 1, 2, 3]:
        axs[row, col].set_aspect('equal')
        month_index = row * 2 + (col // 2)
        if month_index >= 12:
            continue

        if col % 2 == 0:  # SCARIBOS
            speed = speed_regg.isel(time=month_index)
            pcm = axs[row, col].pcolormesh(
                speed.lon, speed.lat, speed,
                cmap=cmo.cm.speed, vmin=vmin, vmax=vmax, rasterized=True
            )
            u = u_regg.isel(time=month_index)
            v = v_regg.isel(time=month_index)
            axs[row, col].quiver(
                u.lon, v.lat,
                u, v,
                color="black", scale=quiver_scale, scale_units="inches", width=quiver_width)
            axs[row, col].text(-70.3, 10.1, "SCARIBOS", fontsize=11, ha='left', va='bottom')

            axs[row, col].set_xlim(-70.5, -66)
            axs[row, col].set_ylim(10, 13.5)

        else:  # GlobCurrent
            speed = speed_all_months.isel(time=month_index)
            pcm = axs[row, col].pcolormesh(
                speed.longitude, speed.latitude, speed,
                cmap=cmo.cm.speed, vmin=vmin, vmax=vmax, rasterized=True
            )
            uo = ds_globcur.uo.isel(time=month_index, depth=0).sel(longitude=lon_slice, latitude=lat_slice)
            vo = ds_globcur.vo.isel(time=month_index, depth=0).sel(longitude=lon_slice, latitude=lat_slice)
            axs[row, col].quiver(
                uo.longitude, uo.latitude,
                uo, vo,
                color="black", scale=quiver_scale, scale_units="inches", width=quiver_width
            )
            axs[row, col].text(-70.3, 10.1, "GlobCurrent", fontsize=11, ha='left', va='bottom')

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

    if row == 5:
        for col in range(4):
            axs[row, col].set_xticks(np.arange(-70, -65, 1))
            axs[row, col].set_xticklabels([f"{abs(tick):.0f}° W" for tick in np.arange(-70, -65, 1)], rotation=90, fontsize=11)
            axs[row, col].set_aspect('equal')

cbar_ax = fig.add_subplot(gs[:, 4]) 
cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='vertical')
cbar.set_label("Speed [m/s]", fontsize=11)
cbar.ax.tick_params(labelsize=11)

# Save the figure
plt.savefig("figures/SCARIBOS_vs_GlobCurrent_surface_velocity.png", dpi=300, bbox_inches='tight')

