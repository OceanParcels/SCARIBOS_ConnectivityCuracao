'''
Script to create buffer area around Curacao to calculate coastal connectivity 
based on scenario COASTCON (Intra-island connectivity)
Coastline is divided into 8 zones following the Coral reef report by Waitt Institute (2017):
            Waitt Institute: Marine Scientific Assessment: The state of Curaçao’s coral reefs, 2017. 
Author: vesnaber
Additinal files needed: croco_grd.nc - input grid file
                        from CROCO model (see repository of SCARIBOS)
'''

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
import xarray as xr
import cmocean
import matplotlib.colors as mcolors

# Upload grid from model CROCO (see repository of SCARIBOS)
config     = 'SCARIBOS_V8'   # CROCO model configuration name
path       = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
grid       = xr.open_dataset(path + 'croco_grd.nc')
bathymetry = grid.h.values
landmask   = grid.mask_rho.values
landmask   = np.where(landmask == 0, 1, np.nan)

# Variables for making buffer areas
release_locations = ['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6', 'zone_7', 'zone_8']
buffer_distance   = 0.017
colors            = plt.cm.tab10(np.linspace(0, 1, len(release_locations)))

# Making buffer areas:
# manually excluding some points to make sure the areas do not overlap (!)
buffered_areas = []
for idx, release_location in enumerate(release_locations):
    zone_data = np.load(f'INPUT/release_locs_COASTCON_{release_location}.npy')
    
    if release_location == 'zone_1':
        points = [Point(lon, lat) for lon, lat in zip(zone_data[0], zone_data[1])]
    elif release_location == 'zone_2':
        lon = zone_data[0]
        lon = [round(l, 3) for l in lon]
        lat = zone_data[1]
        lat = [round(l, 3) for l in lat]

        exclude_combinations = {
            (round(12.05337059, 3), round(-68.74694784, 3)),
            (round(12.0484875, 3), round(-68.75194229, 3)),
            (round(12.0484875, 3), round(-68.74694784, 3)),
            (round(12.05337059, 3), round(-68.75194229, 3)),
            (round(12.04360434, 3), round(-68.76692564, 3)),
            (round(12.04360434, 3), round(-68.77192009, 3)),
        }
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lon)) if lon[i] < -68.82})
        mask = np.array([not (lat[i], lon[i]) in exclude_combinations for i in range(len(lat))])
        lon = np.array(lon)
        lat = np.array(lat)
        points = [Point(lon, lat) for lon, lat in zip(lon[mask], lat[mask])]

    elif release_location == 'zone_3':
        lon = zone_data[0]
        lon = [round(l, 3) for l in lon]
        lat = zone_data[1]
        lat = [round(l, 3) for l in lat]
        exclude_combinations = set()
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lon)) if lon[i] > -68.845})
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lon)) if lon[i] < -68.88})
        mask = np.array([not (lat[i], lon[i]) in exclude_combinations for i in range(len(lat))])
        lon = np.array(lon)
        lat = np.array(lat)
        points = [Point(lon, lat) for lon, lat in zip(lon[mask], lat[mask])]

    elif release_location == 'zone_4':
        lon = zone_data[0]
        lon = [round(l, 3) for l in lon]
        lat = zone_data[1]
        lat = [round(l, 3) for l in lat]
        exclude_combinations = set()
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lon)) if lon[i] > -68.91})
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lon)) if lon[i] < -68.985})
        mask = np.array([not (lat[i], lon[i]) in exclude_combinations for i in range(len(lat))])
        lon = np.array(lon)
        lat = np.array(lat)
        points = [Point(lon, lat) for lon, lat in zip(lon[mask], lat[mask])]
    
    elif release_location == 'zone_5':
        lon = zone_data[0]
        lon = [round(l, 3) for l in lon]
        lat = zone_data[1]
        lat = [round(l, 3) for l in lat]
        exclude_combinations = set()
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lat)) if lat[i] < 12.16})
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lon)) if lon[i] < -69.06})
        mask = np.array([not (lat[i], lon[i]) in exclude_combinations for i in range(len(lat))])
        lon = np.array(lon)
        lat = np.array(lat)
        points = [Point(lon, lat) for lon, lat in zip(lon[mask], lat[mask])]
    
    elif release_location == 'zone_6':
        lon = zone_data[0]
        lon = [round(l, 3) for l in lon]
        lat = zone_data[1]
        lat = [round(l, 3) for l in lat]
        exclude_combinations = set()
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lat)) if lat[i] < 12.21})
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lat)) if lat[i] > 12.27})
        mask = np.array([not (lat[i], lon[i]) in exclude_combinations for i in range(len(lat))])
        lon = np.array(lon)
        lat = np.array(lat)
        points = [Point(lon, lat) for lon, lat in zip(lon[mask], lat[mask])]

    elif release_location == 'zone_7':
        lon = zone_data[0]
        lon = [round(l, 3) for l in lon]
        lat = zone_data[1]
        lat = [round(l, 3) for l in lat]
        exclude_combinations = set()
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lat)) if lat[i] < 12.29})
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lat)) if lat[i] > 12.37})
        mask = np.array([not (lat[i], lon[i]) in exclude_combinations for i in range(len(lat))])
        lon = np.array(lon)
        lat = np.array(lat)
        points = [Point(lon, lat) for lon, lat in zip(lon[mask], lat[mask])]
    
    elif release_location == 'zone_8':
        lon = zone_data[0]
        lon = [round(l, 3) for l in lon]
        lat = zone_data[1]
        lat = [round(l, 3) for l in lat]
        exclude_combinations = set()
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lon)) if lon[i] < -69.13})
        exclude_combinations = exclude_combinations.union({(lat[i], lon[i]) for i in range(len(lon)) if lon[i] > -68.765})
        mask = np.array([not (lat[i], lon[i]) in exclude_combinations for i in range(len(lat))])
        lon = np.array(lon)
        lat = np.array(lat)
        points = [Point(lon, lat) for lon, lat in zip(lon[mask], lat[mask])]

    buffered_area = MultiPoint(points).buffer(buffer_distance)
    buffered_areas.append(buffered_area)
    print(f"Buffered area created for {release_location}.")


# plot created buffered areas together with corresponding release locations
plt.figure(figsize=(10, 10))
for idx, buffered_area in enumerate(buffered_areas):
    plt.fill(*buffered_area.exterior.xy, color=colors[idx], alpha=0.5, label=release_locations[idx])
plt.pcolormesh(grid.lon_rho, grid.lat_rho, landmask, cmap='gray', alpha=1)
plt.title('Buffered Areas')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.xlim(-69.2, -68.6)
plt.ylim(11.9, 12.5)
for idx, release_location in enumerate(release_locations):
    zone_data = np.load(f'INPUT/release_locs_COASTCON_{release_location}.npy')
    plt.scatter(zone_data[0], zone_data[1], 5, color=colors[idx], label=release_location)
plt.legend()

# save buffered areas to npy file
np.save('COASTCON/buffer_areas/COASTCON_buffered_areas.npy', buffered_areas)

