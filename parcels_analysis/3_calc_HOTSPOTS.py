'''
Script to calculate probability density functions (PDF) for the HOTSPOTS scenario
and storing the results as NetCDF files and figures (seperate figures of each month)
parcels scenario name: HOTSPOTS
'''

# import os
import numpy as np
import os
import xarray as xr
import matplotlib.pyplot as plt
import cmocean

# List of months to process
sim_months_list = [
    'Y2020M04', 'Y2020M05', 'Y2020M06', 'Y2020M07', 'Y2020M08', 'Y2020M09', 'Y2020M10', 'Y2020M11', 'Y2020M12',
    'Y2021M01', 'Y2021M02', 'Y2021M03', 'Y2021M04', 'Y2021M05', 'Y2021M06', 'Y2021M07', 'Y2021M08', 'Y2021M09', 'Y2021M10', 'Y2021M11', 'Y2021M12',
    'Y2022M01', 'Y2022M02', 'Y2022M03', 'Y2022M04', 'Y2022M05', 'Y2022M06', 'Y2022M07', 'Y2022M08', 'Y2022M09', 'Y2022M10', 'Y2022M11', 'Y2022M12',
    'Y2023M01', 'Y2023M02', 'Y2023M03', 'Y2023M04', 'Y2023M05', 'Y2023M06', 'Y2023M07', 'Y2023M08', 'Y2023M09', 'Y2023M10', 'Y2023M11', 'Y2023M12',
    'Y2024M01', 'Y2024M02'
]

resolution    = 100
res           = str(resolution)
analysis_type = 'unique'  # 'original'
path          = "OUT_HOTSPOTS/"  # Parcels output path
config        = 'SCARIBOS_V8'  # Model configuration (croco)
path_grd      = '~/croco/CONFIG/' + config + '/CROCO_FILES/'  # Path of grid file
part_config   = 'HOTSPOTS'  # Name of scenario

probability_path       = f"HOTSPOTS/PDF_{analysis_type}_results/"  # Calculated PDFs save path
probability_plots_path = f"HOTSPOTS/PDF_{analysis_type}_figures/"  # Figures save path
os.makedirs(probability_path, exist_ok=True)
os.makedirs(probability_plots_path, exist_ok=True)

# Load grid data
ds_grid   = xr.open_dataset(os.path.expanduser(path_grd + 'croco_grd.nc'))
land_mask = ds_grid.mask_rho.values
land_mask = np.where(land_mask == 0, 1, 0)

# Define region of interest and grid resolution
xmin, xmax     = -69.49, -68.49
ymin, ymax     = 11.67, 12.67
bins_x, bins_y = resolution, resolution

# Cut land_mask to the region of interest
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

# Process each month
for month in sim_months_list:
    print(f"Processing {month}...")

    # Load particles data
    ds = xr.open_zarr(path + part_config + "_" + month + ".zarr")
    lon_particles = ds.lon.values
    lat_particles = ds.lat.values

    # number of all particles
    n_particles = lon_particles.shape[0]
    print(f"Number of particles: {n_particles}")

    # Initialize the matrix for unique particle counts
    unique_particles = np.zeros((bins_y, bins_x))
    unique_particles_norm = np.zeros((bins_y, bins_x))

    # Process each particle trajectory
    for traj_idx in range(lon_particles.shape[0]):
        particle_lon = lon_particles[traj_idx, :]
        particle_lat = lat_particles[traj_idx, :]

        # Skip trajectories with NaN values
        valid_indices = ~np.isnan(particle_lon) & ~np.isnan(particle_lat)

        # Filter for particles within the area of interest
        within_area = (particle_lon >= xmin) & (particle_lon <= xmax) & \
                    (particle_lat >= ymin) & (particle_lat <= ymax)

        # Combine validity and area of interest filters
        valid_indices = valid_indices & within_area
        particle_lon = particle_lon[valid_indices]
        particle_lat = particle_lat[valid_indices]

        # Map particle positions to grid indices
        x_indices = np.digitize(particle_lon, land_mask_cut.lon.values) - 1
        y_indices = np.digitize(particle_lat, land_mask_cut.lat.values) - 1

        # Clip indices to ensure they fall within grid bounds
        x_indices = np.clip(x_indices, 0, bins_x - 1)
        y_indices = np.clip(y_indices, 0, bins_y - 1)

        # Create a temporary matrix for the current particle
        temp_matrix = np.zeros((bins_y, bins_x))
        for x, y in zip(x_indices, y_indices):
            temp_matrix[y, x] = 1  # Mark presence in the grid cell

        # Add the temporary matrix to the unique_particles matrix
        unique_particles += temp_matrix

    # normalize bby dividing with the number of particles
    unique_particles_norm = unique_particles / n_particles

    # Plot the heatmap of unique particles
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cmocean.cm.matter
    pcm = ax.pcolormesh(
        land_mask_cut.lon, land_mask_cut.lat, unique_particles, cmap=cmap, shading='auto'
    )
    cbar = fig.colorbar(pcm, ax=ax, extend='both')
    cbar.set_label('Unique Particles')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Heatmap of Unique Particles ({month})')
    land_mask_plot = np.ma.masked_where(land_mask_cut == 0, land_mask_cut)
    ax.pcolormesh(
        land_mask_cut.lon, land_mask_cut.lat, land_mask_plot, cmap='gray', shading='auto'
    )

    # Save the heatmap figure
    plt.savefig(probability_plots_path + f"heatmap_unique_particles_x{res}y{res}_{month}.png", dpi=300)
    plt.close()

    # Save the unique_particles matrix
    np.save(probability_path + f"unique_particles_x{res}y{res}_{month}.npy", unique_particles)
    np.save(probability_path + f"unique_particles_norm_x{res}y{res}_{month}.npy", unique_particles_norm)

    print(f"Finished processing {month}.")


