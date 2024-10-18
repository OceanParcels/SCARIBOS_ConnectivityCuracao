'''
Comparison of wlev elevation between SCARIBOS and insitu data from Bullenbaai
Needed to run this script:
- insitu data from Bullenbaai (you cna find them in data/tides_bullenbaai_jan2024.txt) 
    --> the data was extracted from the tide gauge found in https://www.ioc-sealevelmonitoring.org/station.php?code=bull 
- SCARIBOS model output for January and February 2024
'''

#%% Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import xarray as xr
import cmocean.cm as cm
from utide import solve
from utide.harmonics import FUV
from utide.utilities import loadbunch
from utide.utilities import Bunch

#%% Load the data - bullenbaai 
# note data is in UTC 

# Load the insitu data
bullentides = pd.read_csv('data/tides_bullenbaai_jan2024.txt', sep=' ', header=None)
bullentides_bubble = bullentides[3].values

bullentides_avg        = np.mean(bullentides_bubble)
bullentides_vals       = bullentides_bubble - bullentides_avg
bullentides_date       = bullentides[0].values
bullentides_time       = bullentides[1].values
bullentides_time_all   = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in bullentides_date+' '+bullentides_time]

# bullentides_time_all has irregular sampling. We need to resample it to 1 hour
bullentides_time_all = pd.to_datetime(bullentides_time_all)
bullentides_time_all = pd.Series(bullentides_vals, index=bullentides_time_all)
bullentides_time_all = bullentides_time_all.resample('1H').mean()
# series to np array:
bullentides_FINAL = bullentides_time_all.values
bullentides_TIME  = bullentides_time_all.index


#%% Load the model data Y2024M01 and M02 - SCARIBOS
dir   = '/nethome/berto006/croco/CONFIG/SCARIBOS_V7/CROCO_FILES/'
file  = f"{dir}croco_his_Y2024M01.nc"
ds    = xr.open_dataset(file)
file2 = f"{dir}croco_his_Y2024M02.nc"
ds2   = xr.open_dataset(file2)

#%% extract time and zeta and combine both

time_jan = ds.time.values
time_feb = ds2.time.values

# Origin time (1st January 2000)
origin_time = pd.Timestamp('2000-01-01')

timedeltas  = pd.to_timedelta(time_jan, unit='s')
timedeltas2 = pd.to_timedelta(time_feb, unit='s')
model_times_jan = origin_time + timedeltas
model_times_feb = origin_time + timedeltas2
index_1jan  = np.where(model_times_jan == pd.Timestamp('2024-01-01 00:00:00'))[0][0]
index_30jan = np.where(model_times_feb == pd.Timestamp('2024-01-30 23:00:00'))[0][0]

# Target location - where the tide gauge is located
target_lon   = -69.0196139
target_lat   = 12.1873306
distance     = np.sqrt((ds.lon_rho - target_lon) ** 2 + (ds.lat_rho - target_lat) ** 2)
min_dist_idx = distance.argmin(dim=['eta_rho', 'xi_rho'])
zeta_jan_all = ds.zeta
zeta_feb_all = ds2.zeta
zeta_jan     = zeta_jan_all.isel(eta_rho=min_dist_idx['eta_rho'], xi_rho=min_dist_idx['xi_rho'])
zeta_feb     = zeta_feb_all.isel(eta_rho=min_dist_idx['eta_rho'], xi_rho=min_dist_idx['xi_rho'])
zeta_jan     = zeta_jan.values[index_1jan:]
zeta_feb     = zeta_feb.values[:index_30jan]

# combine both
modeltides_FINAL = np.concatenate((zeta_jan, zeta_feb), axis=0)
modeltides_FINAL = modeltides_FINAL - np.mean(modeltides_FINAL)
modeltides_TIME  = np.concatenate((model_times_jan[index_1jan:], model_times_feb[:index_30jan]), axis=0)


#%%
# compare constituents

constituents = ['M2', 'S2', 'N2', 'K1', 'O1', 'P1', 'Q1', 'K2', 'M4', 'M6']
# Insitu:
time = bullentides_TIME
tide = bullentides_FINAL
coef_insitu = solve(time, tide, lat=12.1873306, method='ols', conf_int='MC', constit=constituents)
# Model:
time = modeltides_TIME
tide = modeltides_FINAL
coef_model = solve(time, tide, lat=12.1873306, method='ols', conf_int='MC', constit=constituents)

# Extract amplitudes and phases for each constituent
amp_insitu = []
amp_model = []
phase_insitu = []
phase_model = []

for constituent in constituents:
    # Find index for the constituent
    index_insitu = np.where(coef_insitu['name'] == constituent)[0]
    index_model = np.where(coef_model['name'] == constituent)[0]
    
    # Get amplitude and phase
    amp_insitu.append(coef_insitu['A'][index_insitu][0])
    amp_model.append(coef_model['A'][index_model][0])
    phase_insitu.append(coef_insitu['g'][index_insitu][0])
    phase_model.append(coef_model['g'][index_model][0])

# Convert lists to arrays for easier plotting
amp_insitu = np.array(amp_insitu)
amp_model = np.array(amp_model)
phase_insitu = np.array(phase_insitu)
phase_model = np.array(phase_model)


#%% PLOT

color_insitu = 'royalblue'
color_scarib = 'darkorange'
linew_box = 0.2

plt.figure(figsize=(11, 7))

# Time series comparison
plt.subplot(3, 1, 1)
ax = plt.gca()
ax.plot(bullentides_TIME, bullentides_FINAL, label='In-situ Bullenbaai', color=color_insitu)
ax.plot(modeltides_TIME, modeltides_FINAL, label='SCARIBOS model', color=color_scarib)
ax.set_xlabel('Dates [year: 2024]')
plt.xticks(rotation=45)
ax.grid(axis='y', linestyle='--')
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
ax.grid(axis='x', linestyle='--')
ax.set_ylabel('Water level [m]')
ax.set_title('(a) Hourly water level time series')
ax.set_xlim([bullentides_TIME[0], bullentides_TIME[-1]])
ax.legend()
ax.set_ylim([-0.4, 0.4])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_linewidth(linew_box)

# Amplitude comparison
plt.subplot(3, 1, 2)
plt.grid(axis='y', linestyle='--', zorder=0)
plt.bar(np.arange(len(constituents)) - 0.2, amp_insitu, width=0.4, label='In-situ Bullenbaai', color=color_insitu, zorder=3)
plt.bar(np.arange(len(constituents)) + 0.2, amp_model, width=0.4, label='SCARIBOS model', color=color_scarib, zorder=3)
plt.xticks(np.arange(len(constituents)), constituents)
plt.ylabel('Amplitude [m]')
plt.title('(b) Tidal Amplitude')
plt.legend(loc='upper left')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_linewidth(linew_box)

# Phase comparison
plt.subplot(3, 1, 3)
plt.grid(axis='y', linestyle='--', zorder=0)
plt.bar(np.arange(len(constituents)) - 0.2, phase_insitu, width=0.4, label='In-situ Bullenbaai', color=color_insitu, zorder=3)
plt.bar(np.arange(len(constituents)) + 0.2, phase_model, width=0.4, label='SCARIBOS model', color=color_scarib, zorder=3)
plt.xticks(np.arange(len(constituents)), constituents)
plt.ylabel('Phase [°]')
plt.title('(c) Tidal Phase')
plt.legend()
plt.ylim([0, 300])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_linewidth(linew_box)

plt.tight_layout()

# save
plt.savefig('figures/tides_comparison_Jan2024.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/tides_comparison_Jan2024.pdf', dpi=300, bbox_inches='tight')

