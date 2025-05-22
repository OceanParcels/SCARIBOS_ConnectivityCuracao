'''
Script for Supplimentary Figure Fig. S1
In this script we calculate the average surface EKE of the entire domain for the first 6 months. 
Then we plot time series of this EKE.
'''

#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import xroms
import xcmocean
import cmocean.cm as cmo
import pandas as pd
import matplotlib.dates as mdates

# Configuration
config = 'SCARIBOS_V8'
months_year1 = [f'Y2019M{str(m).zfill(2)}' for m in range(12, 13)] + [f'Y2020M{str(m).zfill(2)}' for m in range(1, 13)]
months_year2 = [f'Y2021M{str(m).zfill(2)}' for m in range(1, 13)]

# Function to calculate EKE averages
def calculate_eke_avg(months):
    eke_avg_list = []
    for sim_month in months:
        path = f'/nethome/berto006/croco/CONFIG/{config}/CROCO_FILES/croco_avg_{sim_month}.nc'

        # Load data with xroms
        ds = xroms.open_netcdf(path, chunks={'time': 1})
        ds, xgrid = xroms.roms_dataset(ds, include_cell_volume=True)
        ds.xroms.set_grid(xgrid)

        # Calculate geostrophic velocities and EKE
        ug, vg = xroms.uv_geostrophic(ds.zeta, ds.f, xgrid)
        eke = xroms.EKE(ug, vg, xgrid)

        # Average EKE over the entire domain
        eke_avg = eke.mean(dim=['xi_rho', 'eta_rho'])
        eke_avg_list.append(eke_avg)

    # Concatenate EKE averages across all months
    eke_avg_all = xr.concat(eke_avg_list, dim='time')
    return eke_avg_all.compute()

# Calculate EKE for year 1 and year 2
eke_avg_all_year1 = calculate_eke_avg(months_year1)
eke_avg_all_year2 = calculate_eke_avg(months_year2)

# Calculate moving averages
eke_avg_all_ma_year1 = eke_avg_all_year1.rolling(time=12, center=True).mean().compute()
eke_avg_all_ma_year2 = eke_avg_all_year2.rolling(time=12, center=True).mean().compute()

# Convert time to datetime
time_year1 = pd.to_datetime(eke_avg_all_year1.time.values, unit='s', origin='2000-01-01')
time_year2 = pd.to_datetime(eke_avg_all_year2.time.values, unit='s', origin='2000-01-01')

#%%
# Plot time series of EKE
fig, axs = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

# Year 1 plot
axs[0].plot(time_year1, eke_avg_all_year1, label='EKE (raw)', color='black')
axs[0].plot(time_year1, eke_avg_all_ma_year1, label='EKE (moving average with 12h window)', color='red', linewidth=1.5)
axs[0].set_title('Surface EKE (December 2019 - December 2020)')
axs[0].set_ylabel('EKE [m$^2$/s$^2$]')
axs[0].legend()
axs[0].grid()
axs[0].set_xlim(time_year1[0], time_year1[-1])

# Year 2 plot
axs[1].plot(time_year2, eke_avg_all_year2, label='EKE (raw)', color='black')
axs[1].plot(time_year2, eke_avg_all_ma_year2, label='EKE (moving average with 12h window)', color='red', linewidth=1.5)
axs[1].set_title('Surface EKE (2021)')

axs[1].set_ylabel('EKE [m$^2$/s$^2$]')
axs[1].legend()
axs[1].grid()
axs[1].set_xlim(time_year2[0], time_year2[-1])


axs[0].set_ylim(0, 35)
axs[1].set_ylim(0, 35)

# Year 1 plot ticks (specific dates)
ticks_year1 = pd.date_range(start="2019-12-01", end="2020-12-01", freq='MS') 
axs[0].set_xticks(ticks_year1)
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y')) 
plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)

# Year 2 plot ticks (monthly ticks)
ticks_year2 = pd.date_range(start="2021-01-01", end="2021-12-01", freq='MS') 
axs[1].set_xticks(ticks_year2)
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y')) 
plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45) 

plt.tight_layout()

# save figure
plt.savefig('figures/EKE_timeseries.png', dpi=300)

