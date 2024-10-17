'''
Script to set up the particle release locations for the HOTSPOTS scenario.
Particles will be releasedwithin a square region around Curaçao.
Author: vesnaber
Additinal files needed: croco_grd.nc - input grid file
                        from CROCO model (see repository of SCARIBOS)
'''

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean

# Upload grid from model CROCO (see repository of SCARIBOS)
part_config = 'HOTSPOTS' # Parcels scenario name
config      = 'SCARIBOS_V8' # CROCO model configuration name
path        = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
grid        = xr.open_dataset(path + 'croco_grd.nc')
bathymetry  = grid.h.values
land        = np.where(grid.mask_rho == 0, 1, np.nan)
land        = np.where(grid.mask_rho == 1, np.nan, land)

# Boarders of the square region around Curaçao
xmin = -69.49
xmax = -68.49
ymin = 11.67
ymax = 12.67

# Plot the boarders of the square region around Curaçao
plt.figure(figsize=(10, 10))
plt.pcolormesh(grid.lon_rho, grid.lat_rho, bathymetry, cmap=cmocean.cm.deep)
plt.colorbar(label='Depth [m]', orientation='horizontal')
plt.title('Bathymetry, ' + config)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.pcolormesh(grid.lon_rho, grid.lat_rho, land, cmap=cmocean.cm.gray)
plt.plot([xmin, xmax, xmax, xmin, xmin], 
         [ymin, ymin, ymax, ymax, ymin], 'r-', linewidth=2)
plt.plot(-68.99, 12.17, 'ro') # middle of the island

# Define the grid spacing for the release locations
dx = 0.01
dy = 0.01

# Determine the indices corresponding to the region of interest
xmin_idx = np.where(grid.lon_rho[0, :] >= xmin)[0][0]
xmax_idx = np.where(grid.lon_rho[0, :] <= xmax)[0][-1]
ymin_idx = np.where(grid.lat_rho[:, 0] >= ymin)[0][0]
ymax_idx = np.where(grid.lat_rho[:, 0] <= ymax)[0][-1]

# Crop the land variable to include only the region of interest
cropped_land         = land[ymin_idx:ymax_idx-1, xmin_idx:xmax_idx]
cropped_land_reverse = cropped_land
cropped_land         = np.where(cropped_land == 1, np.nan, 1)

# Create the grid of points
x    = np.arange(xmin, xmax, dx)
y    = np.arange(ymin, ymax, dy)
X, Y = np.meshgrid(x, y)

# Apply the mask to X and Y
X_masked = np.ma.masked_array(X, mask=np.isnan(cropped_land))
Y_masked = np.ma.masked_array(Y, mask=np.isnan(cropped_land))

# Get the indices of unmasked elements
lon_indices, lat_indices = np.where(~X_masked.mask)

# Extract lon and lat values where the mask is False
lon_masked = X_masked[~X_masked.mask].data
lat_masked = Y_masked[~Y_masked.mask].data

# Reshape the lon and lat arrays to match the number of unmasked elements
lon_masked = lon_masked.reshape(len(lon_indices))
lat_masked = lat_masked.reshape(len(lat_indices))

# Plot the grid points excluding land areas
plt.figure(figsize=(6, 6))
plt.pcolormesh(grid.lon_rho, grid.lat_rho, bathymetry, cmap=cmocean.cm.deep)
plt.colorbar(label='Depth [m]', orientation='horizontal')
plt.pcolormesh(grid.lon_rho[ymin_idx:ymax_idx-1, xmin_idx:xmax_idx], grid.lat_rho[ymin_idx:ymax_idx-1, xmin_idx:xmax_idx], cropped_land_reverse, cmap=cmocean.cm.amp)
plt.scatter(X_masked, Y_masked, color='k', marker='.', s=5)
plt.xlabel('Longitude [°]')
plt.ylabel('Latitude [°]')
plt.title('Scenario 1: HOTSPOTS')
plt.xlim(xmin-5*dx, xmax+5*dx)
plt.ylim(ymin-5*dy, ymax+5*dy)
plt.tight_layout()
plt.savefig('figures/INPUT/release_locs_' + part_config + '.png', dpi=300)

# Save the grid of points as python array
np.save('INPUT/release_locs_' + part_config, [lon_masked, lat_masked])

