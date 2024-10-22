'''
Script to process and plot surface currents - monthly averages
Author: vesnaber
Needed to run the script:
- SCARIBOS output files: croco_avg_{month}.nc 
- *OR* if already run script 2_avg_surface_ALLYEARS.py, then the averages are already calcualted and stored in data/ folder
With this script you also save the average surface flow for each month in a separate file
--> used for plotting (so that the calculations do not have to be re-made)
NOTE: The script will automatically skip the calculation of the averges and load the existing files.
'''

#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import cartopy.crs as ccrs
import calendar

# Configuration name
config = 'SCARIBOS_V8'

# List of simulation months - non-existing months are also here as dummies (will be white (invisible) in the plot)
sim_months = ['Y2020M01', 'Y2020M02', 'Y2020M03', 'Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
                'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
                'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
                'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
                'Y2024M01', 'Y2024M02', 'Y2024M03', 'Y2024M04', 'Y2024M05', 'Y2024M06', 'Y2024M07', 'Y2024M08', 'Y2024M09', 'Y2024M10', 'Y2024M11', 'Y2024M12']

xlim = [-69.5, -68.5]
ylim = [11.65, 12.65]

# Create subplots for each month
fig, axes = plt.subplots(5, 12, figsize=(24, 10), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(hspace=0.05, wspace=0.05)

# List of months
month_labels = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

for i, sim_month in enumerate(sim_months):
    year = int(sim_month[1:5])       # Extract year from sim_month
    month_idx = int(sim_month[6:8])  # Extract month index

    # if years are 2024 and months from 4-12, take the Y2024M03 data
    if year == 2024 and month_idx > 3:
        sim_month = 'Y2024M03'
    
    # Define output directory for the current month
    output_dir = f'../CONFIG/{config}/CROCO_FILES/surface_currents/{sim_month}'
    u_file = f'{output_dir}_u.nc'
    v_file = f'{output_dir}_v.nc'
    speed_file = f'{output_dir}_speed.nc'

    # Check if the files already exist
    if os.path.exists(u_file) and os.path.exists(v_file) and os.path.exists(speed_file):
        # Load the files if they exist
        u_top_mean = xr.open_dataset(u_file).u
        v_top_mean = xr.open_dataset(v_file).v
        speed = xr.open_dataset(speed_file).__xarray_dataarray_variable__
        print(f"Files for {sim_month} exist. Loaded u, v, and speed.")
    else:
        # Load CROCO output file for the current month if not already processed
        his_file = f'../CONFIG/{config}/CROCO_FILES/surface_currents/croco_avg_{sim_month}_surface.nc'
        ds_scarib = xr.open_dataset(his_file)
        print(f"Processing {sim_month}")

        # Load and process data for the current month
        u = ds_scarib.u
        v = ds_scarib.v

        # Calculate squared velocity components
        u_top = u
        v_top = v
        sq_u = u_top**2
        sq_v = v_top**2

        # Adjust coordinates for grid alignment
        sq_u.coords["xi_u"] = sq_u.coords["xi_u"] - 0.5
        sq_u.coords["eta_rho"] = sq_u.coords["eta_rho"] + 0.5
        sq_v.coords["xi_rho"] = sq_v.coords["xi_rho"]
        sq_v.coords["eta_v"] = sq_v.coords["eta_v"]
        sq_u = sq_u.rename({"xi_u": "xi_rho", "eta_rho": "eta_v"})

        # Calculate the total squared speed and mean speed over time
        add_sq = sq_u + sq_v
        speed = np.sqrt(add_sq).mean("time")

        # Calculate mean u and v components over time
        u_top_mean = u_top.mean("time")
        v_top_mean = v_top.mean("time")

        # Drop unnecessary dimensions from speed
        speed = speed.drop(["lon_v", "lat_v"])

        # Save u, v, and speed for the current month
        u_top_mean.to_netcdf(u_file)
        v_top_mean.to_netcdf(v_file)
        speed.to_netcdf(speed_file)
        print(f"Saved u, v, and speed for {sim_month}")


    # Mask zero values
    speed_masked = speed.where(speed != 0, other=np.nan)
    print('speed is masked')

    if year == 2020 and month_idx < 4:
    # Replace all values with NaNs by matching the shape of each DataArray
        speed_masked.values = np.full(speed_masked.shape, np.nan)
        u_top_mean.values = np.full(u_top_mean.shape, np.nan)
        v_top_mean.values = np.full(v_top_mean.shape, np.nan)

    # make the same for if year is 2024 and month is greater than 3
    if year == 2024 and month_idx > 3:
    # Replace all values with NaNs by matching the shape of each DataArray
        speed_masked.values = np.full(speed_masked.shape, np.nan)
        u_top_mean.values = np.full(u_top_mean.shape, np.nan)
        v_top_mean.values = np.full(v_top_mean.shape, np.nan)
    
    # Plot speed
    cmap = cmocean.cm.speed
    ax = axes[(i // 12) % 5, i % 12]
    image = speed_masked.plot(x="lon_u", y="lat_u", cmap=cmap, ax=ax, transform=ccrs.PlateCarree(),
                              add_colorbar=False, vmin=0, vmax=1.33, add_labels=False, rasterized=True)
    print('speed is plotted')
    # Customize plot
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add quiver plot (arrows for flow direction)
    skip = (slice(None, None, 16), slice(None, None, 16))  # Controls density of arrows
    ax.quiver(u_top_mean.lon_u[skip], u_top_mean.lat_u[skip], u_top_mean[skip], v_top_mean[skip], transform=ccrs.PlateCarree(), color="black", scale=3, width=0.0093)
    
    # Grid adjustments
    xl = ax.gridlines(draw_labels=False)
    xl.top_labels = False
    xl.right_labels = False
    xl.bottom_labels = False
    xl.left_labels = False
    xl.xlines = False
    xl.ylines = False

    # Add month labels to the top row
    if (i // 12) == 0:
        ax.set_title(month_labels[i % 12], fontsize=20)

    # Add year label to the leftmost column
    if (i % 12) == 0:
        ax.text(-0.1, 0.5, year, va='center', ha='right', fontsize=20, transform=ax.transAxes, rotation=90)

    if year == 2020 and month_idx < 4:
        ax.set_axis_off()

    if year == 2024 and month_idx > 3:
        ax.set_axis_off()

# Title
fig.suptitle("(B) Monthly average current speed and direction around Curaçao", fontsize=22)

# Save the plot as PNG and PDF
plt.savefig(f"figures/{config}_avg_surface_MONTHLY_HQ.png", bbox_inches="tight", dpi=300)
plt.savefig(f"figures/{config}_avg_surface_MONTHLY_HQ.pdf", bbox_inches="tight", dpi=300)


#%%

