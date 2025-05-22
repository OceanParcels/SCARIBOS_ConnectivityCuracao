'''
Script to plot connectivity matrix of REGIOCON scenario (Coastal connectivity)
Needed to run the script:
- calculated connectivity matrix as .npy files, made with 3_calc_REGIOCON.py (for each region and each month/year)
Author: vesnaber
'''


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import cmocean as cmo
from matplotlib.colors import Normalize, ListedColormap
import fitz

# Load the connectivity matrix for each region and year separtely
data_folder = 'REGIOCON/connectivity/REGIOCON_' # location of matrix output

# Aruba:
ARUB_2020 = np.load(data_folder + 'source_ARUB_2020.npy')
ARUB_2021 = np.load(data_folder + 'source_ARUB_2021.npy')
ARUB_2022 = np.load(data_folder + 'source_ARUB_2022.npy')
ARUB_2023 = np.load(data_folder + 'source_ARUB_2023.npy')
ARUB_2024 = np.load(data_folder + 'source_ARUB_2024.npy')
# adding nans to beginning of 2020 and end of 2024 (where there is no data)
ARUB_2020 = np.pad(ARUB_2020, (3, 0),  constant_values=np.nan)
ARUB_2024 = np.pad(ARUB_2024, (0, 10), constant_values=np.nan)

# Bonaire:
BONA_2020 = np.load(data_folder + 'source_BONA_2020.npy')
BONA_2021 = np.load(data_folder + 'source_BONA_2021.npy')
BONA_2022 = np.load(data_folder + 'source_BONA_2022.npy')
BONA_2023 = np.load(data_folder + 'source_BONA_2023.npy')
BONA_2024 = np.load(data_folder + 'source_BONA_2024.npy')
# adding nans to beginning of 2020 and end of 2024 (where there is no data)
BONA_2020 = np.pad(BONA_2020, (3, 0),  constant_values=np.nan)
BONA_2024 = np.pad(BONA_2024, (0, 10), constant_values=np.nan)

# Venezuelan islands (VEIS):
VEIS_2020 = np.load(data_folder + 'source_VEIS_2020.npy')
VEIS_2021 = np.load(data_folder + 'source_VEIS_2021.npy')
VEIS_2022 = np.load(data_folder + 'source_VEIS_2022.npy')
VEIS_2023 = np.load(data_folder + 'source_VEIS_2023.npy')
VEIS_2024 = np.load(data_folder + 'source_VEIS_2024.npy')
# adding nans to beginning of 2020 and end of 2024 (where there is no data)
VEIS_2020 = np.pad(VEIS_2020, (3, 0),  constant_values=np.nan)
VEIS_2024 = np.pad(VEIS_2024, (0, 10), constant_values=np.nan)

# Venezuelan mainland (VECO):
VECO_2020 = np.load(data_folder + 'source_VECO_2020.npy')
VECO_2021 = np.load(data_folder + 'source_VECO_2021.npy')
VECO_2022 = np.load(data_folder + 'source_VECO_2022.npy')
VECO_2023 = np.load(data_folder + 'source_VECO_2023.npy')
VECO_2024 = np.load(data_folder + 'source_VECO_2024.npy')
# adding nans to beginning of 2020 and end of 2024 (where there is no data)
VECO_2020 = np.pad(VECO_2020, (3, 0),  constant_values=np.nan)
VECO_2024 = np.pad(VECO_2024, (0, 10), constant_values=np.nan)

# list of all the matrices
ARUB = np.vstack([ARUB_2020, ARUB_2021, ARUB_2022, ARUB_2023, ARUB_2024])
BONA = np.vstack([BONA_2020, BONA_2021, BONA_2022, BONA_2023, BONA_2024])
VEIS = np.vstack([VEIS_2020, VEIS_2021, VEIS_2022, VEIS_2023, VEIS_2024])
VECO = np.vstack([VECO_2020, VECO_2021, VECO_2022, VECO_2023, VECO_2024])

# variables for plotting
years   = ['2020', '2021', '2022', '2023', '2024']
months  = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
max_all = np.nanmax([np.nanmax(ARUB), np.nanmax(BONA), 
                     np.nanmax(VEIS), np.nanmax(VECO)])
cmap = cmo.cm.matter
cmap = cmap.copy()
cmap.set_bad(color='white')  # Set NaN color to white
norm = mcolors.LogNorm(vmin=1, vmax=100)

# Mask zero values
VEIS = np.ma.masked_equal(VEIS, 0)
ARUB = np.ma.masked_equal(ARUB, 0)
BONA = np.ma.masked_equal(BONA, 0)
VECO = np.ma.masked_equal(VECO, 0)

# Figure:
fig, axs = plt.subplots(3, 2, figsize=(12, 6.5), 
                        gridspec_kw={'width_ratios': [1, 1], 
                        'height_ratios': [1, 1, 0.3]})

# Adjust text color based on value (lighter text for higher values)
def add_text(ax, data, norm):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j] * 100
            if not np.isnan(value):
                text_color = 'bisque' if value > 5 else 'black'
                ax.text(j, i, f'{value:.0f}', ha='center', va='center', color=text_color)

# Aruba:
ARUB_forcolbar = axs[0,0].imshow(ARUB * 100, cmap=cmap, norm=norm)
add_text(axs[0, 0], ARUB, norm)
axs[0,0].set_title('(a) Aruba → Curaçao', fontsize=13)
axs[0,0].set_xticks([])
axs[0,0].set_yticks(np.arange(5))
axs[0,0].set_yticklabels(years)

# Bonaire:
axs[1,0].imshow(VEIS * 100, cmap=cmap, norm=norm)
add_text(axs[1, 0], VEIS, norm)
axs[1,0].set_title('(c) Venezuelan Islands → Curaçao', fontsize=13)
axs[1,0].set_xticks(np.arange(12))
axs[1,0].set_xticklabels(months)
axs[1,0].set_yticks(np.arange(5))
axs[1,0].set_yticklabels(years)

# Venezuelan islands:
axs[0,1].imshow(BONA * 100, cmap=cmap, norm=norm)
add_text(axs[0, 1], BONA, norm)
axs[0,1].set_title('(b) Bonaire → Curaçao', fontsize=13)
axs[0,1].set_xticks([])
axs[0,1].set_yticks([])

# Venezuelan mainland:
axs[1,1].imshow(VECO * 100, cmap=cmap, norm=norm)
add_text(axs[1, 1], VECO, norm)
axs[1,1].set_title('(d) Venezuela (mainland) → Curaçao', fontsize=13)
axs[1,1].set_xticks(np.arange(12))
axs[1,1].set_xticklabels(months)
axs[1,1].set_yticks([])

# Gray and dashed boarders:
for ax in axs.flat:
    for spine in ax.spines.values():
        spine.set_color('grey')
        spine.set_linestyle('--')

# for colorbar:
axs[2,0].axis('off')
axs[2,1].axis('off')
cbar = fig.colorbar(ARUB_forcolbar, ax=axs[1,:], orientation='horizontal', 
                    extend='min', fraction=0.01, pad=0.01)
cbar.set_label('% of particles reaching Curaçao', fontsize=12)
cbar.ax.set_position([0, 0.1, 1, 0.03])
ticks = [1, 10, 100]
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{tick}' for tick in ticks])

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()

plt.savefig('coastconnect_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('coastconnect_matrix.pdf', dpi=300, bbox_inches='tight')

# PLOT A MAP with particle trajectories
# Upload grid from model CROCO (see repository of SCARIBOS) - for bathymetry
config = 'SCARIBOS_V8'   # CROCO model configuration name
path = '~/croco/CONFIG/' + config + '/CROCO_FILES/'
grid = xr.open_dataset(path + 'croco_grd.nc')
bathymetry = grid.h.values
landmask = grid.mask_rho.values
landmask = np.where(landmask == 0, 1, np.nan)

# REGIOCON (Coastal connectivity):
ARUB_traje = np.load('../INPUT/release_locs_REGIOCON_ARUB.npy')
BONA_traje = np.load('../INPUT/release_locs_REGIOCON_BONA.npy')
VEIS_traje = np.load('../INPUT/release_locs_REGIOCON_VEIS.npy')
VECO_traje = np.load('../INPUT/release_locs_REGIOCON_VECO.npy')
curacao_coast_points = np.load('../INPUT/coastal_points_CURACAO.npy')

# make buffer area from coastal points of Curaçao
points = [Point(lon, lat) for lon, lat in zip(curacao_coast_points[0], curacao_coast_points[1])]
multi_point = MultiPoint(points)
buffer_distance = 0.05  # Define the buffer distance
buffered_area_REGIO = multi_point.buffer(buffer_distance)
buffered_gdf = gpd.GeoDataFrame(geometry=[buffered_area_REGIO])
# Create a polygon from the shapely geometry of buffered_area_REGIO
polygon = Polygon(list(zip(buffered_area_REGIO.exterior.xy[0], buffered_area_REGIO.exterior.xy[1])), 
                  facecolor='yellow', edgecolor='orange', hatch='///', alpha=0.6,linewidth=1, label='Destination area: Curaçao', zorder=1)

ARUB_batch = np.size(ARUB_traje[0])
BONA_batch = np.size(BONA_traje[0])
VEIS_batch = np.size(VEIS_traje[0])
VECO_batch = np.size(VECO_traje[0])

# bathymetry colorscheme
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
blevels = [-5000,-4500, -4000, -3500, -3000,-2500, -2000, -1500, -1000, -500, 0]   # define levels for plotting (transition of colorbar)
N       = (len(blevels)-1)*2
bathy_cmap   = custom_div_cmap(N, mincol='#3f3f3f', midcol='dimgrey', midcol2='#888888' ,maxcol='w')
levels = 20
vmin = -5000
vmax = 0

# figure settings:
font0 = 14
font1 = 13
font2 = 16
cura_color = 'slateblue'

# FIGURE OF MAP:
frame = 49 # 50th hour of simulation
scatter_size = 1.5

fig,ax =plt.subplots(figsize=(12, 6))

# Top row, spanning both columns
ax.set_title('(a) Example of particle locations and trajectories in April 2020 after 50 hours', fontsize=14)
bathy = ax.contourf(grid.lon_rho, grid.lat_rho, -bathymetry, levels, cmap=bathy_cmap, vmin=-5000, vmax=0, extend='min', rasterized=True)
for c in bathy.collections:
    c.set_rasterized(True)
ax.add_patch(polygon)
dotscoastsize = 4
ax.contourf(grid.lon_rho, grid.lat_rho, landmask, cmap='Greys', alpha=1, zorder=1)
ax.contourf(grid.lon_rho, grid.lat_rho, landmask, cmap='Greys_r', alpha=0.9, zorder=2)
# Remove borders of the plot
for spine in ax.spines.values():
    spine.set_color('grey')

color = 'orange'
ax.scatter(ds_AR.lon.T[frame, 0:ARUB_batch], ds_AR.lat.T[frame, 0:ARUB_batch], color=color, label='Start', s=scatter_size)
if frame > 12:
    ax.plot(ds_AR.lon.T[frame-6:frame, 0:ARUB_batch], ds_AR.lat.T[frame-6:frame, 0:ARUB_batch], linewidth=0.4, color=color, alpha=0.2)
else:
    ax.plot(ds_AR.lon.T[0:frame, 0:ARUB_batch], ds_AR.lat.T[0:frame, 0:ARUB_batch], linewidth=0.4, color=color, alpha=0.2)
for j in range(0, 30):
    if frame >= 12*j:
        ax.scatter(ds_AR.lon.T[frame-12*j, (j-1)*ARUB_batch:j*ARUB_batch], ds_AR.lat.T[frame-12*j, (j-1)*ARUB_batch:j*ARUB_batch], color=color, label='Start', s=scatter_size)
        if frame >= 12*(j+1):
            ax.plot(ds_AR.lon.T[frame-12*j-6:frame-12*j, (j-1)*ARUB_batch:j*ARUB_batch], ds_AR.lat.T[frame-12*j-6:frame-12*j, (j-1)*ARUB_batch:j*ARUB_batch], linewidth=0.4, color=color, alpha=0.2)
        else:
            ax.plot(ds_AR.lon.T[0:frame-12*j, (j-1)*ARUB_batch:j*ARUB_batch], ds_AR.lat.T[0:frame-12*j, (j-1)*ARUB_batch:j*ARUB_batch], linewidth=0.4, color=color, alpha=0.2)

color = 'tomato'
ax.scatter(ds_BO.lon.T[frame, 0:BONA_batch], ds_BO.lat.T[frame, 0:BONA_batch], color=color, label='Start', s=scatter_size)
if frame > 12:
    ax.plot(ds_BO.lon.T[frame-6:frame, 0:BONA_batch], ds_BO.lat.T[frame-6:frame, 0:BONA_batch], linewidth=0.4, color=color, alpha=0.2)
else:
    ax.plot(ds_BO.lon.T[0:frame, 0:BONA_batch], ds_BO.lat.T[0:frame, 0:BONA_batch], linewidth=0.4, color=color, alpha=0.2)
for j in range(0, 30):
    if frame >= 12*j:
        ax.scatter(ds_BO.lon.T[frame-12*j, (j-1)*BONA_batch:j*BONA_batch], ds_BO.lat.T[frame-12*j, (j-1)*BONA_batch:j*BONA_batch], color=color, label='Start', s=scatter_size)
        if frame >= 12*(j+1):
            ax.plot(ds_BO.lon.T[frame-12*j-6:frame-12*j, (j-1)*BONA_batch:j*BONA_batch], ds_BO.lat.T[frame-12*j-6:frame-12*j, (j-1)*BONA_batch:j*BONA_batch], linewidth=0.4, color=color, alpha=0.2)
        else:
            ax.plot(ds_BO.lon.T[0:frame-12*j, (j-1)*BONA_batch:j*BONA_batch], ds_BO.lat.T[0:frame-12*j, (j-1)*BONA_batch:j*BONA_batch], linewidth=0.4, color=color, alpha=0.2)

color = 'dodgerblue'
ax.scatter(ds_VE.lon.T[frame, 0:VEIS_batch], ds_VE.lat.T[frame, 0:VEIS_batch], color=color, label='Start', s=2)
if frame > 12:
    ax.plot(ds_VE.lon.T[frame-6:frame, 0:VEIS_batch], ds_VE.lat.T[frame-6:frame, 0:VEIS_batch], linewidth=0.4, color=color, alpha=0.2)
else:
    ax.plot(ds_VE.lon.T[0:frame, 0:VEIS_batch], ds_VE.lat.T[0:frame, 0:VEIS_batch], linewidth=0.4, color=color, alpha=0.2)
for j in range(0, 30):
    if frame >= 12*j:
        ax.scatter(ds_VE.lon.T[frame-12*j, (j-1)*VEIS_batch:j*VEIS_batch], ds_VE.lat.T[frame-12*j, (j-1)*VEIS_batch:j*VEIS_batch], color=color, label='Start', s=scatter_size)
        if frame >= 12*(j+1):
            ax.plot(ds_VE.lon.T[frame-12*j-6:frame-12*j, (j-1)*VEIS_batch:j*VEIS_batch], ds_VE.lat.T[frame-12*j-6:frame-12*j, (j-1)*VEIS_batch:j*VEIS_batch], linewidth=0.4, color=color, alpha=0.2)
        else:
            ax.plot(ds_VE.lon.T[0:frame-12*j, (j-1)*VEIS_batch:j*VEIS_batch], ds_VE.lat.T[0:frame-12*j, (j-1)*VEIS_batch:j*VEIS_batch], linewidth=0.4, color=color, alpha=0.2)

color = 'lightseagreen'
ax.scatter(ds_VC.lon.T[frame, 0:VECO_batch], ds_VC.lat.T[frame, 0:VECO_batch], color=color, label='Start', s=2)
if frame > 12:
    ax.plot(ds_VC.lon.T[frame-6:frame, 0:VECO_batch], ds_VC.lat.T[frame-6:frame, 0:VECO_batch], linewidth=0.4, color=color, alpha=0.2)
else:
    ax.plot(ds_VC.lon.T[0:frame, 0:VECO_batch], ds_VC.lat.T[0:frame, 0:VECO_batch], linewidth=0.4, color=color, alpha=0.2)
for j in range(0, 30):
    if frame >= 12*j:
        ax.scatter(ds_VC.lon.T[frame-12*j, (j-1)*VECO_batch:j*VECO_batch], ds_VC.lat.T[frame-12*j, (j-1)*VECO_batch:j*VECO_batch], color=color, label='Start', s=2)
        if frame >= 12*(j+1):
            ax.plot(ds_VC.lon.T[frame-12*j-6:frame-12*j, (j-1)*VECO_batch:j*VECO_batch], ds_VC.lat.T[frame-12*j-6:frame-12*j, (j-1)*VECO_batch:j*VECO_batch], linewidth=0.4, color=color, alpha=0.2)
        else:
            ax.plot(ds_VC.lon.T[0:frame-12*j, (j-1)*VECO_batch:j*VECO_batch], ds_VC.lat.T[0:frame-12*j, (j-1)*VECO_batch:j*VECO_batch], linewidth=0.4, color=color, alpha=0.2)

# Add text for each region
ax.text(-69.9, 12.65, 'Aruba', fontsize=font1, color='black', bbox=dict(facecolor='orange', edgecolor='none', alpha=0.7))
ax.text(-68.3, 12.4, 'Bonaire', fontsize=font1, color='black', bbox=dict(facecolor='tomato', edgecolor='none', alpha=0.7))
ax.text(-67.2, 12.2, 'Venezuelan Islands', fontsize=font1, color='black', bbox=dict(facecolor='dodgerblue', edgecolor='none', alpha=0.7))
ax.text(-69.8, 11.2, 'Venezuela (mainland)', fontsize=font1, color='black', bbox=dict(facecolor='lightseagreen', edgecolor='none', alpha=0.7))

# Display map
ax.set_ylim(10.4, 13)
ax.set_xlim(-70.5, -66)

ax.set_xticks(np.arange(-70, -65, 1))
ax.set_yticks(np.arange(11, 14, 1))
ax.tick_params(axis='both', which='major', labelsize=12)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
ax.set_xticklabels(['{:.0f}° W'.format(abs(x)) for x in ax.get_xticks()])
ax.set_yticklabels(['{:.0f}° N'.format(abs(y)) for y in ax.get_yticks()])
ax.set_aspect('equal', 'box')
cbar = fig.colorbar(bathy, ax=ax, orientation='vertical', extend='min', fraction=0.05, pad=0.05)
cbar.set_label('Depth [m]', fontsize=12)

# legend only for Destination area: Curaçao
ax.legend(handles=[polygon], loc='lower left', fontsize=font1, markerscale=6)

# save as png and pdf
plt.savefig('coastconnect_map_t50.png', dpi=300, bbox_inches='tight')
plt.savefig('coastconnect_map_t50.pdf', dpi=300, bbox_inches='tight')

# Combine the two PDFs (top = map, bottom = connectivity matrix) into one figure
def combine_pdfs_on_one_page(pdf_files, output_file):

    width  = 500 
    height = 540 

    pdf_document  = fitz.open()
    combined_page = pdf_document.new_page(width=width, height=height)
    positions     = [(0, 0.1), (0, height / 2)]

    for i, pdf_file in enumerate(pdf_files):
        pdf         = fitz.open(pdf_file)
        source_page = pdf.load_page(0)
        if i == 0:
            scaling_factor = 1
        else:
            scaling_factor = 1
        rect        = source_page.rect
        scaled_rect = fitz.Rect(rect.x0, rect.y0, rect.x0 + rect.width * scaling_factor, rect.y0 + rect.height * scaling_factor)
        combined_page.show_pdf_page(
            fitz.Rect(positions[i][0], positions[i][1], positions[i][0] + width, positions[i][1] + height / 1.98),
            pdf,
            0,
            clip=scaled_rect
        )
        pdf.close()

    pdf_document.save(output_file)
    pdf_document.close()

combine_pdfs_on_one_page(
    [f"coastconnect_map_t50.pdf", f"coastconnect_matrix.pdf"],
    f"fig10_combined.pdf"
)
