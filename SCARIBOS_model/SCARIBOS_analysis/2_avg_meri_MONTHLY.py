"""
Script to process and plot meridional flow cross seciton analysis - matrix of monthly averages
Author: vesnaber
Needed to run the script:
- SCARIBOS output files: croco_avg_{month}.nc
This script needs long toime to process so it can be run as a job submission script.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean

# =========== configuration name
config = 'SCARIBOS_V8'

# List of simulation months
sim_months = ['Y2020M01', 'Y2020M02', 'Y2020M03', 'Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
                'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
                'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
                'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
                'Y2024M01', 'Y2024M02', 'Y2024M03', 'Y2024M04', 'Y2024M05', 'Y2024M06', 'Y2024M07', 'Y2024M08', 'Y2024M09', 'Y2024M10', 'Y2024M11', 'Y2024M12']

# Create subplots for each month
fig, axes = plt.subplots(5, 12, figsize=(24, 10))
fig.subplots_adjust(hspace=0.05, wspace=0.05)

month_labels = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# Loop over simulation months
for i, sim_month in enumerate(sim_months):
    year = int(sim_month[1:5])  # Extract year
    month_idx = int(sim_month[6:8])  # Extract month index
    
    print(f"Processing {sim_month}...")  # Print progress
    if year == 2024 and month_idx > 3: # create dummy plot for last months of 2024 (will be Nans and thereofre white)
        sim_month = 'Y2024M03'
    
    # Load the corresponding NetCDF file
    his_file = f'../CONFIG/{config}/CROCO_FILES/croco_avg_{sim_month}.nc'
    try:
        ds_scarib = xr.open_dataset(his_file)
    except FileNotFoundError:
        print(f"File not found: {his_file}")
        continue

    # Calculate mean meridional velocity
    u_meri = ds_scarib.u.isel(xi_u=150, eta_rho=slice(145, 220)).mean('time')
    print(f"Calculated mean meridional velocity for {sim_month}")

    # Depth calculation
    s_rho = ds_scarib.s_rho
    h_meri = ds_scarib.h.isel(xi_rho=150, eta_rho=slice(145, 220))
    depth = h_meri * s_rho
    depth_bottom = depth.isel(s_rho=0)
    u_meri.coords["depth"] = depth

    # Apply NaNs to missing data (early 2020 and late 2024)
    if year == 2020 and month_idx < 4 or year == 2024 and month_idx > 3:
        u_meri.values = np.nan * np.ones_like(u_meri.values)
        print(f"Applied NaNs for {sim_month}")

    # Plot the meridional velocity
    ax = axes[i // 12, i % 12]
    image = u_meri.plot(x="lat_u", y="depth", ax=ax, add_colorbar=False, vmin=-0.8, vmax=0.8, cmap=cmocean.cm.balance, rasterized=True)
    print(f"Plotted velocity for {sim_month}")

    # Plot depth
    ax.set_xlim([11.42, 12.149])
    ax.set_ylim([-1400, 0])

    # Customize axes appearance
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Add month labels on the top row
    if (i // 12) == 0:
        ax.set_title(month_labels[i % 12], fontsize=20)
    
    # Add year labels on the left column
    if (i % 12) == 0:
        ax.text(-0.1, 0.5, year, va='center', ha='right', fontsize=20, transform=ax.transAxes, rotation=90)

    if year == 2020 and month_idx < 4:
        ax.set_axis_off()
        ax.fill_between(h_meri.lat_rho, depth_bottom, -1400, color='white', alpha=0.4)
    elif year == 2024 and month_idx > 3:
        ax.set_axis_off()
        ax.fill_between(h_meri.lat_rho, depth_bottom, -1400, color='white', alpha=0.4)
    else:
        ax.fill_between(h_meri.lat_rho, depth_bottom, -1400, color='grey', alpha=0.4)
    

# Add plot title
fig.suptitle(f"B) Monthly average zonal velocity at 69\u00b0 W meridional cross-seciton", fontsize=22)

# Save the plot
plt.savefig(f"figures/{config}_avg_meri_MONTHLY.png", dpi=300, bbox_inches="tight")
plt.savefig(f"figures/{config}_avg_meri_MONTHLY.pdf", bbox_inches="tight", dpi=300)

print("Saved plot as PNG and PDF.")
