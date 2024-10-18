"""
Script to extract surface currents from the Pelagia cruise data (PE529) for validation of SCARIBOS model. 
Needed to run the script:
- processed data from the Pelagia cruise (64PE529) in .nc format - the data was processed using CASCADE software. 
    --> The data is available at NIOZ DAS (DOI avialble in the manuscript)
Author: vesnaber
"""

# Import libraries
import xarray as xr
import numpy as np
import datetime
import pandas as pd

# ========== Import stations 
stations = ['017', '019', '021', '026',
            '039', '041', '043', '045', 
            '056', '058', '060', '062',
            '074', '076', '078', '080',
            '088', '090', '092', '097', 
            '099', '101', '103', 
            '114', '116', '118', '120',
            '130', '132', '134', '136']
level    = 0 # surface = level 0


# ======= Variables

# Initialize arrays to store data for all stations
UVEL_all      = []
VVEL_all      = []
LATITUDE_all  = []
LONGITUDE_all = []
TIME_all      = []
speed_all     = []
dir_all       = []

# for station in stations: open array with u, v, dei, speed, time (of the top layer)
for station in stations:
    path       = 'data/64PE529'+station+'_000000_osite_fhv21.nc'
    ds         = xr.open_dataset(path)
    print(ds.info())

    # ======= Upload variables

    # Velocities
    VVEL        = ds.VVEL_ADCP.values
    UVEL        = ds.UVEL_ADCP.values
    WVEL        = ds.WVEL_ADCP.values
    format      = '%Y%m%d%H%M%S'
    time        = ds.DATE_TIME_UTC.values
    time        = time.astype('str')
    time_interm = time.tolist()
    time_interm = list(map(lambda x: datetime.datetime.strptime(x,  '%Y%m%d%H%M%S').strftime("%Y%m%d %H:%M:%S"), time_interm))
    TIME        = pd.to_datetime(pd.Series(time_interm), format="%Y%m%d %H:%M:%S")
    LONGITUDE   = ds.LONGITUDE.values
    LATITUDE    = ds.LATITUDE.values
    LATITUDE    = np.array(LATITUDE)
    LONGITUDE   = np.array(LONGITUDE)

    # ======== Calculate speed and direction
    speed = np.sqrt(UVEL**2 + VVEL**2)
    dir = np.mod(360 + (180/np.pi)* np.arctan2(UVEL, VVEL), 360)
    # Append data to the arrays
    UVEL_all.append(UVEL[:, level])
    VVEL_all.append(VVEL[:, level])
    LATITUDE_all.append(LATITUDE)
    LONGITUDE_all.append(LONGITUDE)
    TIME_all.append(TIME)
    speed_all.append(speed[:, level])
    dir_all.append(dir[:, level])

# ======= Concatenate data from all stations
UVEL_all      = np.concatenate(UVEL_all)
VVEL_all      = np.concatenate(VVEL_all)
LATITUDE_all  = np.concatenate(LATITUDE_all)
LONGITUDE_all = np.concatenate(LONGITUDE_all)
TIME_all      = np.concatenate(TIME_all)
speed_all     = np.concatenate(speed_all)
dir_all       = np.concatenate(dir_all)


# ======= save data
np.savez('data/Pelagia_transects_data.npz',
         UVEL_all=UVEL_all,
         VVEL_all=VVEL_all,
         LATITUDE_all=LATITUDE_all,
         LONGITUDE_all=LONGITUDE_all,
         TIME_all=TIME_all,
         speed_all=speed_all,
         dir_all=dir_all)

