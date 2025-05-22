'''
Type of Parcels scenario: HOTSPOTS (with seeding every 12 hours)
Type of analysis: normalized unique particle count
Script to plot combination of monthly and average over all years
'''

#%%
# Import libraries
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean.cm as cmo
import cartopy
import cartopy.crs as ccrs
import cmocean
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import fitz

resolution       = 100
res              = str(resolution)
type             = 'log'
logmax           = 0.025
logmin           = 0.0025

part_config            = 'HOTSPOTS'
analysis_type          = 'unique' # counting the unique number of particles
path                   = "OUT_HOTSPOTS/"  # Parcels output path
probability_path       = f"HOTSPOTS/PDF_{analysis_type}_results/"  # Calculated PDFs save path
probability_plots_path = f"HOTSPOTS/PDF_{analysis_type}_figures/"  # Figures save path
config                 = 'SCARIBOS_V8'  # Model configuration (croco)
path_grd               = '~/croco/CONFIG/' + config + '/CROCO_FILES/'  # Path of grid file
part_config            = 'HOTSPOTS'  # Name of scenario

sim_months = ['Y2020M01', 'Y2020M02', 'Y2020M03', 'Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
              'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
              'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
              'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
              'Y2024M01', 'Y2024M02', 'Y2024M03', 'Y2024M04', 'Y2024M05', 'Y2024M06', 'Y2024M07', 'Y2024M08', 'Y2024M09', 'Y2024M10', 'Y2024M11', 'Y2024M12'] # this list is just for plotting purpuses

sim_months_true = ['Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
              'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
              'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
              'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
              'Y2024M01', 'Y2024M02'] # this is the list of all simulaiton months

# Load grid data
ds_grid = xr.open_dataset(os.path.expanduser(path_grd + 'croco_grd.nc'))
land_mask = ds_grid.mask_rho.values
land_mask = np.where(land_mask == 0, 1, 0)

# Define region of interest and grid resolution
xmin, xmax = -69.49, -68.49
ymin, ymax = 11.67, 12.67
bins_x, bins_y = resolution, resolution
x1 = np.where(ds_grid.lon_rho.values[0, :] >= xmin)[0][0]
x2 = np.where(ds_grid.lon_rho.values[0, :] <= xmax)[0][-1]
y1 = np.where(ds_grid.lat_rho.values[:, 0] >= ymin)[0][0]
y2 = np.where(ds_grid.lat_rho.values[:, 0] <= ymax)[0][-1]

land_mask_cut = xr.DataArray(
    land_mask[y1:y2, x1:x2],
    dims=["lat", "lon"],
    coords={"lat": ds_grid.lat_rho.values[y1:y2, 0], "lon": ds_grid.lon_rho.values[0, x1:x2]}
)
land_mask_cut = land_mask_cut.interp(
    lat=np.linspace(ymin, ymax, bins_y),
    lon=np.linspace(xmin, xmax, bins_x),
    method="nearest"
)

def check_probability_file(filepath):
    return os.path.exists(filepath)

def probability_map_empty(probability, ax, title):
    """ Plot probability map with logarithmic scale """
    ax.set_aspect('equal', 'box')
    plt.plot(probability, rasterized=True)
    ax.set_axis_off() 
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])

def probability_map_log(probability, ax, title):
    """ Plot probability map with logarithmic scale """
    ax.set_aspect('equal', 'box')
    land_mask_plot = np.ma.masked_where(land_mask_cut == 0, land_mask_cut)

    ax.pcolormesh(
        land_mask_cut.lon, land_mask_cut.lat, probability, cmap=cmo.matter, norm=LogNorm(vmin=logmin, vmax=logmax), shading='auto', rasterized=True
    )
    ax.pcolormesh(
        land_mask_cut.lon, land_mask_cut.lat, land_mask_plot, cmap='Greys', rasterized=True
    )

# total over all years: 
total_probability = np.zeros((bins_y, bins_x))
for sim_month in sim_months_true:
    probability = np.load(probability_path + f"unique_particles_norm_x{res}y{res}_{sim_month}.npy")
    total_probability += probability/len(sim_months)
total_probability = np.ma.masked_where(land_mask_cut.values == 1, total_probability)

#%% FIGURE 1: MONTHLY PROBABILITY MAPS (bottom plot of the final figure)

fig, axes = plt.subplots(5, 12, figsize=(24, 10), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(hspace=0.05, wspace=0.05)

cbar_ax = fig.add_axes([0.125, 0.05, 0.77, 0.025])  # [left, bottom, width, height]

plt.suptitle(f"b) Monthly normalized unique particle count", fontsize=24, y=0.98)
# cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.025])

month_labels = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

for i, sim_month in enumerate(sim_months):
    year = int(sim_month[1:5])
    month_idx = int(sim_month[6:8])
    print(f"Processing {sim_month} for year {year} and month index {month_idx}")

    if year == 2020 and month_idx < 4:
        sim_month = 'Y2020M04'

    if year == 2024 and month_idx > 2:
        sim_month = 'Y2021M08'

    probability = np.load(probability_path + f"unique_particles_norm_x{res}y{res}_{sim_month}.npy")

    print(f"Loaded probability data for {sim_month}")

    ax = axes[i // 12, i % 12]

    if (year == 2020 and month_idx < 4) or (year == 2024 and month_idx > 2):
        probability_nan = np.full((bins_x, bins_y), np.nan)
        probability  = probability_nan
        probability_map_empty(probability, ax, sim_month)

    else:
        probability_map_log(probability, ax, sim_month)
    
    if (i // 12) == 0:
        ax.set_title(month_labels[i % 12], fontsize=20)

    if (i % 12) == 0:
        ax.text(-0.1, 0.5, year, va='center', ha='right', fontsize=20, transform=ax.transAxes, rotation=90)

cbar = plt.colorbar(plt.cm.ScalarMappable(norm=LogNorm(vmin=logmin, vmax=logmax), cmap=cmo.matter), cax=cbar_ax, orientation='horizontal', extend='both')
ticks = np.linspace(logmin, logmax, 6)
ticks = [0.0025, 0.005, 0.01, 0.015, 0.02, 0.025]
cbar.set_ticks(ticks)
tick_labels = [f"{tick:.3f}" for tick in ticks]
tick_labels = [r'$2.5\times10^{-3}$', r'$5\times10^{-3}$', 
               r'$1\times12^{-2}$', r'$1.5\times10^{-2}$', 
               r'$2\times10^{-2}$', r'$2.5\times10^{-2}$']
cbar.ax.set_xticklabels(tick_labels, fontsize=18)
cbar.ax.xaxis.get_offset_text().set_fontsize(18)
cbar.ax.tick_params(labelsize=20)
cbar.set_label('Normalized particle count', fontsize=22)

# Save figure as pdf at first (in order to combine both plots together with fitz package)
plt.savefig(f'HOTSPOTS/UNIQUE_x{res}y{res}_{type}_MONTHLY_PDF_NORM_FINALFORPAPER.pdf', dpi=300, bbox_inches='tight')


# %% FIGURE 2: TOTAL PROBABILITY MAP (top plot of the final figure)

# redefine logmin and logmax for total probability plot
logmin = 0.003
logmax = 0.01

fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={'projection': ccrs.PlateCarree()})
cmap = cmo.matter

# suptitle
plt.suptitle(f"a) Normalized unique particle count, time average over entire period ", fontsize=16, y=0.94, x=0.55)

# Plot the data
pcm = ax.pcolormesh(
    np.linspace(xmin, xmax, bins_x),
    np.linspace(ymin, ymax, bins_y),
    total_probability,
    norm=LogNorm(vmin=logmin, vmax=logmax),
    cmap=cmap,
    rasterized=True
)

cbar_ax = fig.add_axes([0.85, 0.125, 0.02, 0.75])  # [left, bottom, width, height]

cbar = plt.colorbar(pcm, cax=cbar_ax, orientation='vertical', extend='both')
ticks = [0.003, 0.004, 0.006, 0.009, 0.01]
cbar.set_ticks(ticks)  # Set tick positions explicitly

tick_labels = [r'$3\times10^{-3}$', r'$4\times10^{-3}$', r'$6\times10^{-3}$', r'$9\times10^{-3}$', r'$1\times10^{-2}$']
cbar.ax.set_yticklabels(tick_labels, fontsize=12)  # Set the corresponding tick labels
cbar.ax.tick_params(labelsize=12)
cbar.set_label('Normalized particle count', fontsize=14)

# Set axis limits
ax.set_xlim([xmin, xmax-0.005])
ax.set_ylim([ymin, ymax-0.005])

x_ticks = np.arange(xmin, xmax+0.00001, 0.2)
y_ticks = np.arange(ymin+0.1, ymax+0.1, 0.2)
ax.set_xticks(x_ticks)
ax.set_yticks(y_ticks)
ax.tick_params(labelsize=12)

# Format x and y tick labels with degree notation
def format_lon(lon):
    direction = 'E' if lon >= 0 else 'W'
    return f"{abs(lon):.1f}° {direction}"

def format_lat(lat):
    if lat == 0:
        return "0°"
    direction = 'N' if lat > 0 else 'S'
    return f"{abs(lat):.1f}° {direction}"

ax.set_xticklabels([format_lon(tick) for tick in x_ticks], fontsize=12)
ax.set_yticklabels([format_lat(tick) for tick in y_ticks], fontsize=12)

# save
plt.savefig(f'HOTSPOTS/UNIQUE_x{res}y{res}_{type}_ALLYEARS_PDF_NORM_FINALFORPAPER.pdf', dpi=300, bbox_inches='tight')

# %% FINAL FIGURE: merge both pdfs together (top is the total probability map, bottom is the monthly probability maps)

type = 'log' # name of the type of the plot (logarithmic scale used in paper)

def combine_pdfs_on_one_page(pdf_files, output_file):

    width  = 500 
    height = 540 

    pdf_document  = fitz.open()
    combined_page = pdf_document.new_page(width=width, height=height)
    positions     = [(0, 0), (0, height / 2)]

    for i, pdf_file in enumerate(pdf_files):
        pdf         = fitz.open(pdf_file)
        source_page = pdf.load_page(0)
        if i == 0:
            scaling_factor = 1.2
        else:
            scaling_factor = 1.2
        rect        = source_page.rect
        scaled_rect = fitz.Rect(rect.x0, rect.y0, rect.x0 + rect.width * scaling_factor, rect.y0 + rect.height * scaling_factor)
        combined_page.show_pdf_page(
            fitz.Rect(positions[i][0], positions[i][1], positions[i][0] + width, positions[i][1] + height / 2),
            pdf,
            0,
            clip=scaled_rect
        )
        pdf.close()

    pdf_document.save(output_file)
    pdf_document.close()

combine_pdfs_on_one_page(
    [f"HOTSPOTS/UNIQUE_x{res}y{res}_{type}_ALLYEARS_PDF_NORM_FINALFORPAPER.pdf", f"HOTSPOTS/UNIQUE_x{res}y{res}_{type}_MONTHLY_PDF_NORM_FINALFORPAPER.pdf"],
    f"HOTSPOTS/FINAL_x{res}y{res}_{type}_COMBINED.pdf"
)

