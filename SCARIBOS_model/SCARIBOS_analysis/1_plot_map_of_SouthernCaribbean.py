"""
Script to plot the bathymetry of the Southern Caribbean with the SCARIBOS model domain
Author: vesnaber
Needed to run the script:
- bathymetry from ETOPO - data/bathy_etopo2.nc
- bathymetry from GEBCO and Pelagia merged - data/gebco_and_pelagia_merged_SCARIBOS_V2.nc
- shape of Curacao as shapefile for plotting (CUW_adm0.shp), found at www.gadm.org, contributor: OCHA Field Information Services Section (FISS), available publicly
"""

#%%
import fitz 
import numpy as np
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap


# Load the shapefile of Curacao
shapefile_path = 'data/CUW_adm0.shp'
land = gpd.read_file(shapefile_path)

# Load the bathymetry data
bathy_gebpel_file = 'data/data_large_files/gebco_and_pelagia_merged_SCARIBOS_V2.nc'
bathy_gebpel = xr.open_dataset(bathy_gebpel_file)
bathy_gebpel_topo = bathy_gebpel['topo']

# Define a custom colormap for the bathymetry
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
blevels = [-5373, -4000, -3500, -3000, -2000, -1500, -1000, -750, -500, -250, 0]   # define levels for plotting (transition of colorbar)
N       = (len(blevels)-1)*2
cmap2_bl   = custom_div_cmap(N, mincol='#3f3f3f', midcol='dimgrey', midcol2='#888888' ,maxcol='w')
levels = 10
vmin = -5373
vmax = 0

# Load the bathymetry data from ETOPO
etopo_data = xr.open_dataset('data/data_large_files/bathy_etopo2.nc')
bathymetry = etopo_data['z']
bathymetry_subregion = bathymetry.sel(latitude=slice(8.5, 16), longitude=slice(-73, -60))

# figure settings
font0 = 12
font1 = 14
square_color = 'darkorange' 
square_cur_color = 'k'
square_linewidth = 2  
square_cur_linewidth = 1

# FIGURE:
fig = plt.figure(figsize=(11, 6))
gs = gridspec.GridSpec(1, 1) 

# Main Plot - Southern Caribbean Bathymetry
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
contourf = ax1.contourf(bathymetry_subregion['longitude'], bathymetry_subregion['latitude'], 
                        bathymetry_subregion, levels, cmap=cmap2_bl, vmin=vmin, vmax=vmax, 
                        transform=ccrs.PlateCarree(), extend='min', rasterized=True)
# for c in contourf.collections:
#     c.set_rasterized(True)
# ax1.coastlines(resolution='50m', color='grey', linewidth=0.5)
ax1.add_feature(cfeature.LAND, zorder=1, color='saddlebrown', alpha=0.4)
gridlines = ax1.gridlines(draw_labels=False, zorder=1, linewidth=0.5)
gridlines.xlabels_top = False
gridlines.ylabels_right = False
gridlines.xlabels_bottom = False
gridlines.ylabels_left = False
ax1.set_yticks(np.arange(8, 18, 1))
ax1.set_xticks(np.arange(-76, -58, 2))
ax1.set_yticklabels(['{:.0f}° N'.format(abs(lat)) for lat in ax1.get_yticks()])
ax1.set_xticklabels(['{:.0f}° W'.format(abs(lon)) for lon in ax1.get_xticks()])

# squares (domain of SCARIBOS and square around Curacao)
ax1.plot([-70.5, -70.5], [10, 13.5], color=square_color, linewidth=square_linewidth, zorder=2)
ax1.plot([-70.5, -66], [10, 10], color=square_color, linewidth=square_linewidth, zorder=2)
ax1.plot([-66, -66], [10, 13.5], color=square_color, linewidth=square_linewidth, zorder=2)
ax1.plot([-70.5, -66], [13.5, 13.5], color=square_color, linewidth=square_linewidth, zorder=2)
ax1.plot([-69.4, -69.4], [11.8, 12.6], color=square_cur_color , linewidth=square_cur_linewidth, zorder=2, linestyle = '--')
ax1.plot([-69.4, -68.5], [11.8, 11.8], color=square_cur_color , linewidth=square_cur_linewidth, zorder=2, linestyle = '--')
ax1.plot([-68.5, -68.5], [11.8, 12.6], color=square_cur_color , linewidth=square_cur_linewidth, zorder=2, linestyle = '--')
ax1.plot([-69.4, -68.5], [12.6, 12.6], color=square_cur_color , linewidth=square_cur_linewidth, zorder=2, linestyle = '--')
ax1.plot([-69.4, -70.1], [12.6, 13.4], color=square_cur_color , linewidth=square_cur_linewidth, zorder=2)

# Inset Plot - Zoomed in on Curacao
curacao_extent = [-69.4, -68.5, 11.8, 12.6]  # [west, east, south, north]
inset_ax = fig.add_axes([0, 0.635, 0.3, 0.3], projection=ccrs.PlateCarree())
inset_ax.set_extent(curacao_extent)
contourf_zoomed = inset_ax.contourf(bathy_gebpel_topo['lon'], bathy_gebpel_topo['lat'], bathy_gebpel_topo, 25, 
                                    cmap=cmap2_bl,vmin=-5000, vmax = 0, rasterized=True)
# for c in contourf_zoomed.collections:
#     c.set_rasterized(True)
inset_ax.add_geometries(land.geometry, crs=ccrs.PlateCarree(), facecolor='saddlebrown',alpha = 0.4,  edgecolor='k', linewidth=0.5, zorder = 2, rasterized=True)

# colorbar
cbar_ax = fig.add_axes([0.923, 0.06, 0.025, 0.888]) 
cbar = plt.colorbar(contourf, cax=cbar_ax, orientation='vertical', label='Depth [m]', shrink=0.9, extend=None)
cbar.set_label('Depth [m]', fontsize=font0)
ticks = [-5000, -4500, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500, 0]
cbar.set_ticks(ticks) 
cbar.ax.tick_params(labelsize=font0)

font9 = 10
ax1.annotate('SCARIBOS model domain', xy=(-69, 13.7), 
             xytext=(-69.2, 13.65), fontsize=font0, 
             color = square_color)
ax1.annotate('Paraguaná', xy=(-69.98, 11.7), 
             xytext=(-69.98, 11.2), fontsize=font9, color = square_cur_color, 
             arrowprops=dict(facecolor='black', arrowstyle='->'))
ax1.annotate('Peninsula', xy=(-70, 11.9), 
             xytext=(-69.98, 11), fontsize=font9, 
             color = square_cur_color)
ax1.annotate('V e n e z u e l a', xy=(-69.5, 11.5), 
             xytext=(-68.3, 9.2), fontsize=18, 
             color = square_cur_color)
ax1.annotate('Aruba', xy=(-69.9, 12.5), 
             xytext=(-70.2, 12.7), fontsize=font9, 
             color = square_cur_color)
ax1.annotate('Curaçao', xy=(-68.9, 12.2), 
             xytext=(-69.37, 12.65), fontsize=font9, 
             color = 'w')
ax1.annotate('Bonaire', xy=(-68.3, 12.2), 
             xytext=(-68.4, 12.42), fontsize=font9, 
             color = 'w')
ax1.annotate('L e s s e r', xy=(-62.3, 13.5), 
             xytext=(-62.3, 13.1), fontsize=18, 
             color = 'w', rotation = 90)
ax1.annotate('A n t i l l e s', xy=(-62.5, 13.5), 
             xytext=(-61.8, 12.9), fontsize=18, 
             color = 'w', rotation = 90)

font9 =9
inset_ax.annotate('Northern Coastline', xy=(-69.1, 11.5), 
                  xytext=(-69.1, 12.15), fontsize=font9, 
                  color = square_cur_color, rotation = -40)
inset_ax.annotate('Southern Coastline', xy=(-68.67, 12.38), 
                  xytext=(-69.3, 11.88), fontsize=font9, 
                  color = square_cur_color, rotation = -40)
inset_ax.annotate('Klein', xy=(-68.65, 11.98), 
                  xytext=(-68.75, 11.88), fontsize=font9,
                  color = square_cur_color, 
                  arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA=0.05, shrinkB=0.05))
inset_ax.annotate('Curaçao', xy=(-68.65, 11.98), 
                  xytext=(-68.75, 11.823), fontsize=font9, 
                  color = square_cur_color)

plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)  # Adjust margins


plt.savefig('bathymetry_southern_caribbean_with_inset.png', dpi=300, bbox_inches='tight')


# %%
