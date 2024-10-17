'''
Script to set up the particle release locations for the COASTCON scenario (= Intra-island connectivity)
(connectivity within the coastal zones around Curaçao).
COastline is divided into 8 zones following the Coral reef report by Waitt Institute (2017):
            Waitt Institute: Marine Scientific Assessment: The state of Curaçao’s coral reefs, 2017. 
Author: vesnaber, with help from Michael Denes
Additinal files needed: croco_grd.nc - input grid file
                        from CROCO model (see repository of SCARIBOS)
'''

#%% Set up coastal points

# Import libraries:
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cmocean
from glob import glob
import sys
from scipy.ndimage import zoom

# Upload grid from model CROCO (see repository of SCARIBOS)
part_config = 'COASTCON'    # Parcels scenario name
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

# Find land of Curaçao
lons       = grid.lon_rho.values
lats       = grid.lat_rho.values
mask_land  = land
mask_coast = get_mask_coast(mask_land, lons, lats)

# Increase the resolution of mask_coast by a factor of 2 (you can change this factor as needed)
factor           = 2
mask_coast_finer = interpolate_grid(mask_coast, factor)
lons_finer       = interpolate_grid(lons, factor)
lats_finer       = interpolate_grid(lats, factor)
coast_indices    = np.where(mask_coast_finer > 0.5)
coast_lon        = lons_finer[coast_indices]
coast_lat        = lats_finer[coast_indices]

# Plot coastal points to check if they are correct
plt.figure(figsize=(10, 10))
plt.pcolormesh(grid.lon_rho, grid.lat_rho, bathymetry, cmap=cmocean.cm.deep)
plt.colorbar(label='Depth [m]', orientation='horizontal')
plt.title('Bathymetry, ' + config)
plt.scatter(coast_lon, coast_lat, color='red', marker='.', label='Coastal Points', s=5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Coastal Points with Bathymetry Background')
plt.legend()

# Take only the coastal points of Curaçao
coast_lon_1 = coast_lon[(coast_lon > -69.5) & (coast_lon < -68.5) & (coast_lat > 11.7) & (coast_lat < 12.5)]
coast_lat_1 = coast_lat[(coast_lon > -69.5) & (coast_lon < -68.5) & (coast_lat > 11.7) & (coast_lat < 12.5)]

# Plot area around Curaçao
plt.figure(figsize=(10, 10))
plt.pcolormesh(grid.lon_rho, grid.lat_rho, bathymetry, cmap=cmocean.cm.deep)
plt.colorbar(label='Depth [m]', orientation='horizontal')
plt.title('Bathymetry, ' + config)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('COASTCON - particle release locations')
plt.scatter(coast_lon_1, coast_lat_1, color='red', marker='.', label='Coastal Points 1', s=10)
plt.legend()
plt.xlim(-69.3, -68.6)
plt.ylim(11.85, 12.5)
plt.grid()

#%% Divide coastal points into 8 zones:
# Manually chosen points based on the map from the Waitt Institute (2017)

# Zone 1: Klein Curacao
zone1_lon = coast_lon_1[0:36]
zone1_lat = coast_lat_1[0:36]

# Zone 2: Oostpunt
zone2_lon = coast_lon_1[36:77]
zone2_lat = coast_lat_1[36:77]

# Zone 3: Caracasbaai
zone3_lon = coast_lon_1[(coast_lon_1 > -68.895) & (coast_lon_1 < -68.82) & (coast_lat_1 > 12.055) & (coast_lat_1 < 12.1)]
zone3_lat = coast_lat_1[(coast_lon_1 > -68.895) & (coast_lon_1 < -68.82) & (coast_lat_1 > 12.055) & (coast_lat_1 < 12.1)]

# Zone 4: Willemstad
zone4_lon = coast_lon_1[(coast_lon_1 > -69.0) & (coast_lon_1 < -68.895) & (coast_lat_1 > 12.055) & (coast_lat_1 < 12.146)]
zone4_lat = coast_lat_1[(coast_lon_1 > -69.0) & (coast_lon_1 < -68.895) & (coast_lat_1 > 12.055) & (coast_lat_1 < 12.146)]

# Zone 5: Bullenbaai
zone5_lon = coast_lon_1[(coast_lon_1 > -69.075) & (coast_lon_1 < -68.995) & (coast_lat_1 > 12.1) & (coast_lat_1 < 12.2)]
zone5_lat = coast_lat_1[(coast_lon_1 > -69.075) & (coast_lon_1 < -68.995) & (coast_lat_1 > 12.1) & (coast_lat_1 < 12.2)]

# Zone 6: Valentijnsbaai
zone6_lon = coast_lon_1[(coast_lon_1 > -69.145) & (coast_lon_1 < -69.074) & (coast_lat_1 > 12.18) & (coast_lat_1 < 12.31)]
zone6_lat = coast_lat_1[(coast_lon_1 > -69.145) & (coast_lon_1 < -69.074) & (coast_lat_1 > 12.18) & (coast_lat_1 < 12.31)]

# Zone 7: Westpunt
zone7_lon = coast_lon_1[(coast_lon_1 > -69.2) & (coast_lon_1 < -69.145) & (coast_lat_1 > 12.2) & (coast_lat_1 < 12.4)]
zone7_lat = coast_lat_1[(coast_lon_1 > -69.2) & (coast_lon_1 < -69.145) & (coast_lat_1 > 12.2) & (coast_lat_1 < 12.4)]

# Zone 8: North Shore
# making sure no replicas are made
zone1_indices = set(range(0, 36))
zone2_indices = set(range(36, 77))
zone3_indices = set(np.where((coast_lon_1 > -68.895) & (coast_lon_1 < -68.82) & (coast_lat_1 > 12.055) & (coast_lat_1 < 12.1))[0])
zone4_indices = set(np.where((coast_lon_1 > -69.0) & (coast_lon_1 < -68.895) & (coast_lat_1 > 12.055) & (coast_lat_1 < 12.146))[0])
zone5_indices = set(np.where((coast_lon_1 > -69.075) & (coast_lon_1 < -68.995) & (coast_lat_1 > 12.1) & (coast_lat_1 < 12.2))[0])
zone6_indices = set(np.where((coast_lon_1 > -69.145) & (coast_lon_1 < -69.074) & (coast_lat_1 > 12.18) & (coast_lat_1 < 12.31))[0])
zone7_indices = set(np.where((coast_lon_1 > -69.2) & (coast_lon_1 < -69.145) & (coast_lat_1 > 12.2) & (coast_lat_1 < 12.4))[0])
all_indices = set(range(len(coast_lon_1)))
used_indices = zone1_indices.union(zone2_indices).union(zone3_indices).union(zone4_indices).union(zone5_indices).union(zone6_indices).union(zone7_indices)
zone8_indices = all_indices - used_indices
zone8_lon = coast_lon_1[list(zone8_indices)]
zone8_lat = coast_lat_1[list(zone8_indices)]

# Count the number of particles in each zone
print('Zone 1: Klein Curacao: ', len(zone1_lon))
print('Zone 2: Oostpunt: ', len(zone2_lon))
print('Zone 3: Caracasbaai: ', len(zone3_lon))
print('Zone 4: Willemstad: ', len(zone4_lon))
print('Zone 5: Bullenbaai: ', len(zone5_lon))
print('Zone 6: Valentijnsbaai: ', len(zone6_lon))
print('Zone 7: Westpunt: ', len(zone7_lon))
print('Zone 8: North Shore: ', len(zone8_lon))

# plot these coastlines with different colors
plt.figure(figsize=(8, 8))
plt.pcolormesh(grid.lon_rho, grid.lat_rho, bathymetry, cmap=cmocean.cm.deep)
plt.colorbar(label='Depth [m]', orientation='horizontal')
plt.title('Bathymetry, ' + config)
plt.xlabel('Longitude [°]')
plt.ylabel('Latitude [°]')
plt.title('Particle release locations: Scenario 2')
plt.scatter(zone1_lon, zone1_lat, color='purple', marker='.', label='Zone 1: Klein Curacao', s=20)
plt.scatter(zone2_lon, zone2_lat, color='blue', marker='.', label='Zone 2: Oostpunt', s=20)
plt.scatter(zone3_lon, zone3_lat, color='green', marker='.', label='Zone 3: Caracasbaai', s=20)
plt.scatter(zone4_lon, zone4_lat, color='pink', marker='.', label='Zone 4: Willemstad', s=20)
plt.scatter(zone5_lon, zone5_lat, color='orange', marker='.', label='Zone 5: Bullenbaai', s=20)
plt.scatter(zone6_lon, zone6_lat, color='red', marker='.', label='Zone 6: Valentijnsbaai', s=20)
plt.scatter(zone7_lon, zone7_lat, color='darkred', marker='.', label='Zone 7: Westpunt', s=20)
plt.scatter(zone8_lon, zone8_lat, color='black', marker='.', label='Zone 8: North Shore', s=20)
plt.legend()
plt.xlim(-69.3, -68.6)
plt.ylim(11.9, 12.45)
plt.savefig('figures/INPUT/release_locs_' + part_config + '.png', dpi=300)

#%% Saving the release locations for Parcels input

np.save('INPUT/release_locs_'+ part_config+'_zone_1' , [zone1_lon, zone1_lat])
np.save('INPUT/release_locs_'+ part_config+'_zone_2' , [zone2_lon, zone2_lat])
np.save('INPUT/release_locs_'+ part_config+'_zone_3' , [zone3_lon, zone3_lat])
np.save('INPUT/release_locs_'+ part_config+'_zone_4' , [zone4_lon, zone4_lat])
np.save('INPUT/release_locs_'+ part_config+'_zone_5' , [zone5_lon, zone5_lat])
np.save('INPUT/release_locs_'+ part_config+'_zone_6' , [zone6_lon, zone6_lat])
np.save('INPUT/release_locs_'+ part_config+'_zone_7' , [zone7_lon, zone7_lat])
np.save('INPUT/release_locs_'+ part_config+'_zone_8' , [zone8_lon, zone8_lat])

