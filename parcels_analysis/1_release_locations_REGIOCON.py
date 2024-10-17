'''
Script to set up the particle release locations for the REGIOCON scenario (= Coastal connectivity)
Coastal zones are divided into coastline of Aruba, Bonaire, Venezuelan islands and portion of Venezuelan mainland.
Author: vesnaber, with help from Michael Denes
Additinal files needed: croco_grd.nc - input grid file
                        from CROCO model (see repository of SCARIBOS)
'''

#%%
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean
from glob import glob
import sys
from scipy.ndimage import zoom

# Upload grid from model CROCO (see repository of SCARIBOS)
part_config = 'REGIOCON'    # Parcels scenario name
config     = 'SCARIBOS_V8'  # CROCO model configuration name
path       = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
grid       = xr.open_dataset(path + 'croco_grd.nc')
bathymetry = grid.h.values
land       = np.where(grid.mask_rho == 0, 1, 0)


# Function to calculate the coastal mask
def get_mask_coast(mask_land, lons, lats, outfile='./tmp_mask_coast'):
    '''
    calculate the coast mask. With coastal cells, we mean cells in the water, adjacent to land
    '''
    if os.path.exists(outfile):
        ds = xr.open_dataset(outfile)
        mask_coast = np.array(ds['mask_coast'],dtype=bool)    
    else:
        #check the upper,lower,left & right neighbor: if one of these is an ocean cell, set to landborder
        mask_coast = ~mask_land & (np.roll(mask_land,1,axis=0) | np.roll(mask_land,-1,axis=0) | 
                                       np.roll(mask_land,1,axis=1) | np.roll(mask_land,-1,axis=1))
        
    return mask_coast


def interpolate_grid(data, factor):
    """
    Interpolate a 2D grid to a higher resolution using scipy.ndimage.zoom.
    
    Parameters:
    data (numpy array): 2D array to be interpolated.
    factor (int): Factor by which to increase resolution.
    
    Returns:
    numpy array: Interpolated 2D array.
    """
    return zoom(data, factor, order=1)  # Use bilinear interpolation


lons = grid.lon_rho.values
lats = grid.lat_rho.values
mask_land = land
mask_coast = get_mask_coast(mask_land, lons, lats)

# Increase the resolution of mask_coast by a factor of 2 (you can change this factor as needed)
factor = 2
mask_coast_finer = interpolate_grid(mask_coast, factor)
lons_finer = interpolate_grid(lons, factor)
lats_finer = interpolate_grid(lats, factor)
coast_indices = np.where(mask_coast_finer > 0.5)
coast_lon = lons_finer[coast_indices]
coast_lat = lats_finer[coast_indices]


# Plot coastal points to check if they are correct
plt.figure(figsize=(10, 10))
plt.pcolormesh(grid.lon_rho, grid.lat_rho, bathymetry, cmap=cmocean.cm.deep)
plt.colorbar(label='Depth [m]', orientation='horizontal')
plt.title('Bathymetry, ' + config)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(coast_lon, coast_lat, color='red', marker='.', label='Coastal Points', s=5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Coastal Points with Bathymetry Background')
plt.legend()
plt.tight_layout()



#%% Divide coast into Aruba, Bonaire, Venezuelan islands and portion of Venezuelan mainland
# this is done manually to carefully select the correct coastal points

# part 1: curacao - save for making a buffer zone (for post-processing)
coast_lon_1 = coast_lon[(coast_lon > -69.5) & (coast_lon < -68.5) & (coast_lat > 11.7) & (coast_lat < 12.5)]
coast_lat_1 = coast_lat[(coast_lon > -69.5) & (coast_lon < -68.5) & (coast_lat > 11.7) & (coast_lat < 12.5)]

# part 2: aruba
coast_lon_2 = coast_lon[(coast_lon > -70.5) & (coast_lon < -69.5) & (coast_lat > 12.25) & (coast_lat < 13)]
coast_lat_2 = coast_lat[(coast_lon > -70.5) & (coast_lon < -69.5) & (coast_lat > 12.25) & (coast_lat < 13)]

# part 3: bonaire
coast_lon_3 = coast_lon[(coast_lon > -68.5) & (coast_lon < -68) & (coast_lat > 11.7) & (coast_lat < 12.5)]
coast_lat_3 = coast_lat[(coast_lon > -68.5) & (coast_lon < -68) & (coast_lat > 11.7) & (coast_lat < 12.5)]

# part 4: venezuelan islands
coast_lon_4 = coast_lon[(coast_lon > -68) & (coast_lon < -66) & (coast_lat > 11.5) & (coast_lat < 13)]
coast_lat_4 = coast_lat[(coast_lon > -68) & (coast_lon < -66) & (coast_lat > 11.5) & (coast_lat < 13)]

# ***Venezuelan mainlad - excluding Golfete de Coro
# part 5: venezuelan coast east
coast_lon_5 = coast_lon[(coast_lon > -68.5) & (coast_lon < -66.01) & (coast_lat > 10) & (coast_lat < 11)]
coast_lat_5 = coast_lat[(coast_lon > -68.5) & (coast_lon < -66.01) & (coast_lat > 10) & (coast_lat < 11)]

# part 6: venezuelan coast west
coast_lon_6 = coast_lon[(coast_lon > -69.71) & (coast_lon < -68.2) & (coast_lat > 11) & (coast_lat < 11.65)]
coast_lat_6 = coast_lat[(coast_lon > -69.71) & (coast_lon < -68.2) & (coast_lat > 11) & (coast_lat < 11.65)]
coast_lon_6_part2 = coast_lon[(coast_lon > -69.76) & (coast_lon < -69.71) & (coast_lat > 11.6) & (coast_lat < 11.71)]
coast_lat_6_part2 = coast_lat[(coast_lon > -69.76) & (coast_lon < -69.71) & (coast_lat > 11.6) & (coast_lat < 11.71)]
coast_lon_6 = np.concatenate((coast_lon_6, coast_lon_6_part2))
coast_lat_6 = np.concatenate((coast_lat_6, coast_lat_6_part2))

# part 7: venezuelan coast west - what is left
coast_lon_7 = coast_lon[(coast_lon > -72) & (coast_lon < -69.5) & (coast_lat > 11.71) & (coast_lat < 12.3)]
coast_lat_7 = coast_lat[(coast_lon > -72) & (coast_lon < -69.5) & (coast_lat > 11.71) & (coast_lat < 12.3)]

# part 8: excluding Golfete de Coro
coast_lon_8 = coast_lon[(coast_lon > -72) & (coast_lon < -70.12) & (coast_lat > 11) & (coast_lat < 11.75)]
coast_lat_8 = coast_lat[(coast_lon > -72) & (coast_lon < -70.12) & (coast_lat > 11) & (coast_lat < 11.75)]

# combine coasts 5 to 8 = entire continental venezuela (excluding Golfete de Coro)
coast_lon_5_8 = np.concatenate((coast_lon_5, coast_lon_6, coast_lon_7, coast_lon_8))
coast_lat_5_8 = np.concatenate((coast_lat_5, coast_lat_6, coast_lat_7, coast_lat_8))

# get rid of duplicates
coast_5_8 = np.column_stack((coast_lon_5_8, coast_lat_5_8))
unique_coast_5_8 = np.unique(coast_5_8, axis=0)
coast_lon_5_8 = unique_coast_5_8[:, 0]
coast_lat_5_8 = unique_coast_5_8[:, 1]

# plot
plt.figure(figsize=(10, 10))
plt.pcolormesh(grid.lon_rho, grid.lat_rho, bathymetry, cmap=cmocean.cm.deep)
plt.colorbar(label='Depth [m]', orientation='horizontal')
plt.title('Bathymetry, ' + config)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('SCARIPAS - coastal release')
plt.scatter(coast_lon_1, coast_lat_1, color='red', marker='.', label='Curacao', s=5)
plt.scatter(coast_lon_2, coast_lat_2, color='blue', marker='.', label='Aruba', s=5)
plt.scatter(coast_lon_3, coast_lat_3, color='green', marker='.', label='Bonaire', s=5)
plt.scatter(coast_lon_4, coast_lat_4, color='purple', marker='.', label='Venezuelan islands', s=5)
plt.scatter(coast_lon_5_8, coast_lat_5_8, color='black', marker='.', label='Venezuelan mainland', s=5)
plt.legend()
plt.grid()
plt.savefig('figures/release_locs_' + part_config + '.png', dpi=300)

# Save the release locations
np.save('INPUT/coastal_points_CURACAO', [coast_lon_1, coast_lat_1])         # Curacao
np.save('INPUT/release_locs_REGIOCON_ARUB', [coast_lon_2, coast_lat_2])     # Aruba
np.save('INPUT/release_locs_REGIOCON_BONA', [coast_lon_3, coast_lat_3])     # Bonaire
np.save('INPUT/release_locs_REGIOCON_VEIS', [coast_lon_4, coast_lat_4])     # Venezuelan islands
np.save('INPUT/release_locs_REGIOCON_VECO', [coast_lon_5_8, coast_lat_5_8]) # Venezuelan mainland


# %%
