'''
Script to plot hotspots as logharitmic probability density functions (PDF)
NOTE: PDFs are calculated with the script: 3_calc_HOTSPOTS.py (!)
'''

# Import libraries
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cmocean
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

# variables
probability_path = "HOTSPOTS/PDF_unfiltered/" # here the calculated PDFs are stored
part_config      = 'HOTSPOTS'
xmin, xmax       = -69.49, -68.49 # zooming in around Curaçao
ymin, ymax       = 11.67, 12.67
bins_x, bins_y   = 600, 500 # same bins as in 3_calc_HOTSPOTS.py, to create dummy plots (empty months)

# set up colorbar limits to detect hotspots 
logmin = 4.999e-4 
logmax = 5.001e-3

# List of months
sim_months = ['Y2020M01', 'Y2020M02', 'Y2020M03', 'Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
              'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
              'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
              'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
              'Y2024M01', 'Y2024M02', 'Y2024M03', 'Y2024M04', 'Y2024M05', 'Y2024M06', 'Y2024M07', 'Y2024M08', 'Y2024M09', 'Y2024M10', 'Y2024M11', 'Y2024M12']

# Check if probability data files exist
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
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal', 'box')
    ax.add_feature(cartopy.feature.OCEAN, zorder=0, facecolor='white') 
    ax.add_feature(cartopy.feature.LAND, zorder=1, facecolor='white', edgecolor='black')
    probability.plot(ax=ax, cmap=cmo.matter, norm=LogNorm(vmin=logmin, vmax=logmax), add_colorbar=False, rasterized=True)
    ax.set_rasterization_zorder(2)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])


# Figure
fig, axes = plt.subplots(5, 12, figsize=(24, 10), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# Initialize colorbar
cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.025])

month_labels = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

for i, sim_month in enumerate(sim_months):
    year = int(sim_month[1:5])
    month_idx = int(sim_month[6:8])
    print(f"Processing {sim_month} for year {year} and month index {month_idx}")

    if year == 2020 and month_idx < 4:
        sim_month = 'Y2020M04'

    if year == 2024 and month_idx > 2:
        sim_month = 'Y2024M01'
    
    probability_file = f'{probability_path}{part_config}_PDF_unfiltered_x600_y500_{sim_month}.nc'

    probability = xr.open_dataset(probability_file)
    print(f"Loaded probability data for {sim_month}")

    ax = axes[i // 12, i % 12]
    if (year == 2020 and month_idx < 4) or (year == 2024 and month_idx > 2):
        # make empthy nan array with the same shape as the probability array (shape is bins_x, bins_y)
        probability_nan = np.full((bins_x, bins_y), np.nan)
        probability  = probability_nan
        probability_map_empty(probability, ax, sim_month)

    else:
        probability_map_log(probability.probability.T, ax, sim_month)
    
    if (i // 12) == 0:
        ax.set_title(month_labels[i % 12], fontsize=20)

    if (i % 12) == 0:
        ax.text(-0.1, 0.5, year, va='center', ha='right', fontsize=20, transform=ax.transAxes, rotation=90)

# Create the colorbar with specified orientation
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=LogNorm(vmin=logmin, vmax=logmax), cmap=cmo.matter), 
                    cax=cbar_ax, orientation='horizontal', extend='both')

# Manually set the ticks to include the values you want
ticks = [5e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
cbar.set_ticks(ticks)

# Set the tick labels manually and in scientific notation
tick_labels = [r'$5\times10^{-4}$', r'$1\times10^{-3}$', r'$2\times10^{-3}$', r'$3\times10^{-3}$', r'$4\times10^{-3}$', r'$5\times10^{-3}$']
cbar.ax.set_xticklabels(tick_labels, fontsize=18)

# Set the offset text size if scientific notation is used
cbar.ax.xaxis.get_offset_text().set_fontsize(18)

# Set the tick label font size
cbar.ax.tick_params(labelsize=20)

# Set colorbar label and font size
cbar.set_label('Probability [%]', fontsize=22)

# Save figure
plt.savefig('figures/HOTSPOTS_monthly_PDF.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/HOTSPOTS_monthly_PDF.pdf', dpi=300, bbox_inches='tight')
