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

plt.savefig('figures/coastconnect_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/coastconnect_matrix.pdf', dpi=300, bbox_inches='tight')