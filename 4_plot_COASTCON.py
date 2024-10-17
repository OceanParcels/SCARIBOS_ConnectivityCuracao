'''
Script to plot hotspots as logharitmic probability density functions (PDF)
for manuscript
Files needed: 
- connectivity calculated in 3_calc_COASTCON.py
- buffered areas for COASTCON scenario, named COASTCON_buffered_areas.npy
- croco_grd.nc - input grid file from CROCO model (see repository of SCARIBOS) - for bathymetry
'''

#%% Import libraries
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cmocean
import cmocean.cm as cmo
import seaborn as sns
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, Normalize
import geopandas as gpd
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# for plotting bathymetry
def custom_div_cmap(numcolors=50, name='custom_div_cmap',
                    mincol='blue', midcol2='yellow', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with three colors
    
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                              colors=[mincol, midcol, midcol2, maxcol],
                                              N=numcolors)
    return cmap
blevels    = [-5000, -4000, -3000, -2500, -2000, -1500, -1000,-800, -600,  -400, -200, 0]   # define levels for plotting (transition of colorbar)
N          = (len(blevels)-1)*2
bathy_cmap = custom_div_cmap(N, mincol='#696969', midcol='dimgrey', midcol2='#888888' ,maxcol='w')
vmin       = -8000
vmax       = 0

# Upload grid from model CROCO (see repository of SCARIBOS) - for bathymetry
config     = 'SCARIBOS_V8'
path       = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
grid       = xr.open_dataset(path + 'croco_grd.nc')
bathymetry = grid.h.values
landmask   = grid.mask_rho.values
landmask   = np.where(landmask == 0, 1, np.nan)
oceanmask  = np.where(landmask == 1, 1, np.nan)
oceanmask  = np.where(landmask == 1, np.nan, 1)

# load buffered areas
buffered_areas    = np.load('COASTCON/buffer_areas/COASTCON_buffered_areas.npy', allow_pickle=True)
release_locations = ['zone_1', 'zone_2', 'zone_3', 'zone_4', 'zone_5', 'zone_6', 'zone_7', 'zone_8']
colors            = plt.cm.ocean_r(np.linspace(0, 1, len(release_locations)+1))

# Load connectivity data (calculate this in 3_calc_COASTCON.py)
data              = {}
source_zones      = [1, 2, 3, 4, 5, 6, 7, 8]  # Assuming 1-8 for 8 source zones
destination_zones = range(8)                  # Assuming 0-7 for 8 destination zones
for source in source_zones:
    for dest in destination_zones:
        file_path = f'COASTCON/connectivity/COASTCON_source_zone_{source}_destination_zone_{dest}.npy'
        data[f'source{source}dest{dest+1}'] = np.load(file_path)

# change dictionary such that the data is accessible as float arrays
for i in range(1, 9):
    exec(f'source1dest{i}_val = data["source1dest{i}"][:, 1].astype(float) * 100')
    exec(f'source2dest{i}_val = data["source2dest{i}"][:, 1].astype(float) * 100')
    exec(f'source3dest{i}_val = data["source3dest{i}"][:, 1].astype(float) * 100')
    exec(f'source4dest{i}_val = data["source4dest{i}"][:, 1].astype(float) * 100')
    exec(f'source5dest{i}_val = data["source5dest{i}"][:, 1].astype(float) * 100')
    exec(f'source6dest{i}_val = data["source6dest{i}"][:, 1].astype(float) * 100')
    exec(f'source7dest{i}_val = data["source7dest{i}"][:, 1].astype(float) * 100')
    exec(f'source8dest{i}_val = data["source8dest{i}"][:, 1].astype(float) * 100')

# source4dest4_100 = np.full(47, 1.0)
# data['source4dest4'] = np.column_stack((data['source4dest4'][:, 0], source4dest4_100))

# Calculate average for all 4 years together:
connectivity_matrix_avg = np.zeros((8, 8, 47))
for i in range(1, 9):
    for j in range(1, 9):
        connectivity_matrix_avg[i-1, j-1]      = data[f'source{i}dest{j}'][:, 1].astype(float) * 100
connectivity_matrix_avg = connectivity_matrix_avg.mean(axis=2)

# Monthly matrix:
nan_array = np.full(47, np.nan)
source4dest4_val = np.full(47, 100)
# Concatenate the source-destination values, and then reshape
connectivity_matrix = np.column_stack((
    source1dest1_val, source1dest2_val, source1dest3_val, source1dest4_val, source1dest5_val, source1dest6_val, source1dest7_val, source1dest8_val, nan_array,
    source2dest1_val, source2dest2_val, source2dest3_val, source2dest4_val, source2dest5_val, source2dest6_val, source2dest7_val, source2dest8_val, nan_array,
    source3dest1_val, source3dest2_val, source3dest3_val, source3dest4_val, source3dest5_val, source3dest6_val, source3dest7_val, source3dest8_val, nan_array,
    source4dest1_val, source4dest2_val, source4dest3_val, source4dest4_val, source4dest5_val, source4dest6_val, source4dest7_val, source4dest8_val, nan_array,
    source5dest1_val, source5dest2_val, source5dest3_val, source5dest4_val, source5dest5_val, source5dest6_val, source5dest7_val, source5dest8_val, nan_array,
    source6dest1_val, source6dest2_val, source6dest3_val, source6dest4_val, source6dest5_val, source6dest6_val, source6dest7_val, source6dest8_val, nan_array,
    source7dest1_val, source7dest2_val, source7dest3_val, source7dest4_val, source7dest5_val, source7dest6_val, source7dest7_val, source7dest8_val, nan_array,
    source8dest1_val, source8dest2_val, source8dest3_val, source8dest4_val, source8dest5_val, source8dest6_val, source8dest7_val, source8dest8_val
))

# add zero rows as an extra row after 9 rows (esthetics)
connectivity_matrix_2020 = connectivity_matrix[:9]
connectivity_matrix_2021 = connectivity_matrix[9:21]
connectivity_matrix_2022 = connectivity_matrix[21:33]
connectivity_matrix_2023 = connectivity_matrix[33:45]
connectivity_matrix_2024 = connectivity_matrix[45:]

# Create a row of NaN with the shape of (1, 71)
nan_row = np.full((1, 71), np.nan)

# Stack the matrices with NaN rows in between
connectivity_matrix = np.vstack((
    connectivity_matrix_2020, nan_row,
    connectivity_matrix_2021, nan_row,
    connectivity_matrix_2022, nan_row,
    connectivity_matrix_2023, nan_row,
    connectivity_matrix_2024
))

# if zero, replace with 0.00000001 (to avoid log(0))
connectivity_matrix[connectivity_matrix == 0] = 0.00000001


#%% Plotting:

fig = plt.figure(figsize=(15, 16))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.7], width_ratios=[0.14, 1, 1.1, 0.1])
fig.subplots_adjust(hspace=5)
fig.subplots_adjust(wspace=0)

# PLOT 1: Map of zones
axs1 = fig.add_subplot(gs[0, 1])
bathy = axs1.contourf(grid.lon_rho, grid.lat_rho, -bathymetry, 60, cmap=bathy_cmap, vmin=-8000, vmax=0)
for c in bathy.collections:
    c.set_rasterized(True)
for idx, buffered_area in enumerate(buffered_areas):
    axs1.fill(*buffered_area.exterior.xy, color=colors[idx+1], alpha=0.8, label=str(idx + 1))
axs1.set_aspect('equal', 'box')
axs1.contourf(grid.lon_rho, grid.lat_rho, landmask, cmap='Greys', alpha=1, zorder=1)
axs1.contourf(grid.lon_rho, grid.lat_rho, landmask, cmap='Greys_r', alpha=0.9, zorder=2)
axs1.set_title('(a) Map of zones', fontsize=16)
axs1.set_xticks(np.arange(-69.2, -68.6, 0.2))
axs1.set_yticks(np.arange(12, 12.6, 0.2))
axs1.tick_params(axis='both', which='major', labelsize=13)
axs1.set_xticklabels(['{:.1f}°W'.format(abs(x)) for x in axs1.get_xticks()])
axs1.set_yticklabels(['{:.1f}°N'.format(abs(y)) for y in axs1.get_yticks()])
axs1.set_xlim(-69.21, -68.52)
axs1.set_ylim(11.94, 12.55)
for idx, release_location in enumerate(release_locations):
    zone_data = np.load(f'INPUT/release_locs_COASTCON_{release_location}.npy')
    axs1.scatter(zone_data[0], zone_data[1], 3, color=colors[idx + 1])
axs1.legend(loc='upper right', fontsize=14, title=None)
axs1.legend(['1: Klein Curaçao', '2: Oostpunt', '3: Caracasbaai', '4: Willemstad', '5: Bullenbaai', '6: Valentijnsbaai', '7: Westpunt', '8: North Shore'], loc='upper right', fontsize=14)
axs1.spines['top'].set_color('white')
axs1.spines['right'].set_color('white')
axs1.spines['bottom'].set_color('white')
axs1.spines['left'].set_color('white')

# PLOT 2: Average connectivity
destinations = [f'{i}' for i in range(1, 9)]
axs2 = fig.add_subplot(gs[0, 2])
heatmap = sns.heatmap(connectivity_matrix_avg, xticklabels=destinations, yticklabels=destinations, annot=True,
                      cmap='cmo.matter', cbar=False, norm=mcolors.LogNorm(1, 100), fmt=".0f", ax=axs2)
axs2.set_title('(b) Average connectivity (2020-2024)', fontsize=16)
axs2.set_xlabel('Destination Zones', fontsize=14)
axs2.set_ylabel('Source Zones', fontsize=14)
axs2.tick_params(axis='x', labelsize=14)  # Adjust the size of the x-tick labels
axs2.tick_params(axis='y', labelsize=14)

# PLOT 3: Monthly connectivity
axs3 = fig.add_subplot(gs[1, :])  # Span across both columns
masked_matrix = np.ma.masked_invalid(connectivity_matrix)
cmap = plt.get_cmap('cmo.matter').copy()
cmap.set_bad(color='white')  # This
norm = mcolors.LogNorm(vmin=1, vmax=100)
axs2.set_aspect('equal', 'box')
im = axs3.imshow(masked_matrix, cmap=cmap, norm=norm)
axs3.set_xlabel('Destination Zones', fontsize=14)

# colorbar:
cbar = fig.colorbar(im, ax=axs2, orientation='vertical', extend='min')
cbar.set_label('Connectivity [%]', fontsize=14)
cbar.set_ticks([1, 10, 100])
cbar.ax.set_yticklabels(['1', '10', '100'])
cbar.ax.tick_params(labelsize=14)

# White boundary of the plot
plt.gca().spines['top'].set_color('white')
plt.gca().spines['right'].set_color('white')
plt.gca().spines['bottom'].set_color('white')
plt.gca().spines['left'].set_color('white')

labels = ['1', '2', '3', '4', '5', '6', '7', '8', ' ',
          '1', '2', '3', '4', '5', '6', '7', '8', ' ',
          '1', '2', '3', '4', '5', '6', '7', '8', ' ',
          '1', '2', '3', '4', '5', '6', '7', '8', ' ',
          '1', '2', '3', '4', '5', '6', '7', '8', ' ',
          '1', '2', '3', '4', '5', '6', '7', '8', ' ',
          '1', '2', '3', '4', '5', '6', '7', '8', ' ',
          '1', '2', '3', '4', '5', '6', '7', '8']
month_labels = ['2020:   A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D', 
                '2021:    J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D', 
                '2022:    J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D', 
                '2023:    J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D', 
                '2024:    J', 'F']
month_boundaries = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50]
axs3.set_yticks(month_boundaries)
axs3.set_yticklabels(month_labels, fontsize=12, rotation=0)

positions = [i for i, label in enumerate(labels) if label != ' ']
filtered_labels = [label for label in labels if label != ' ']
axs3.set_xticks(positions)
axs3.set_xticklabels(filtered_labels, fontsize=14)

fig.text(0.5, 0.6115, '(c) Monthly connectivity', ha='center', fontsize=16)
fig.text(0.5, 0.595, 'Source Zones', ha='center', fontsize=14)
startright = 0.148
toright = 0.1
totop = 0.581
fig.text(startright,             totop, '1', ha='center', fontsize=14)
fig.text(startright + toright,   totop, '2', ha='center', fontsize=14)
fig.text(startright + 2*toright, totop, '3', ha='center', fontsize=14)
fig.text(startright + 3*toright, totop, '4', ha='center', fontsize=14)
fig.text(startright + 4*toright, totop, '5', ha='center', fontsize=14)
fig.text(startright + 5*toright, totop, '6', ha='center', fontsize=14)
fig.text(startright + 6*toright, totop, '7', ha='center', fontsize=14)
fig.text(startright + 7*toright, totop, '8', ha='center', fontsize=14)

plt.tight_layout()

plt.savefig('figures/COASTCON_all_for_MS.png', dpi=300)
plt.savefig('figures/COASTCON_all_for_MS.pdf', dpi=300)
# plt.savefig('figures/fig08.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('figures/fig08.png', dpi=300, bbox_inches='tight')