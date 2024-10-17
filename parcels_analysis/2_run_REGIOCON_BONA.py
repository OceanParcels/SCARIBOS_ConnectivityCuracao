"""
Script to run the Parcels simulation for the REGIOCON scenario (= Coastal connectivity)
This script runs in parallel for each year and each month, as specified in the .sh file. 
Release location: BONA = Bonaire
Needed files: release locaitons of particles, made with 1_release_locations_REGIOCON.py
"""

import sys
import numpy as np
from glob import glob
import xarray as xr
from parcels import *
from parcels import StatusCode
import datetime
from datetime import datetime, timedelta
import calendar

location = 'BONA'

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>; if you see this message, you forgot to add the year and/or month as arguments!")
        sys.exit(1)
    
    year  = int(sys.argv[1])
    month = int(sys.argv[2])
        
    # chosen variables
    part_config   = 'REGIOCON'
    seeding_dt    = 12      # in hours - particles are seeded every seeding_dt hours
    seeding_start = 0       # in hours - particles are seeded at seeding_start hours (to prevent overlap of previous releases)
    adv_period    = 30      # in days, particles are advected for the period of adv_period days (they are deleted after that)
    ipart_dt      = 5*60    # seconds

    print(f"Start running 2_run_{part_config}_{location}.py...")
    print("Simulation: SCARIBOS_V8 croco model, releasing particles in the coastal grid every 1/100 degree (approx)")
    print(f"Particles are released every {seeding_dt} hours and are advected for {adv_period} days (if None, then it means 365 days or less if the year ends before that)")
    print('Particle configuration name: '+part_config)
    print("Python kernel used: parcels")
    print("Parcels kernels used: AdvectionRK4, FreezeParticle")

    # Define the base directory - where hydrodynamic data is stored (SCARIBOS)
    dir = '/nethome/berto006/croco/CONFIG/SCARIBOS_V8/CROCO_FILES/surface_currents/'

    # generate the list of hydrodynamic data filenames, starting from
    # current month until the end of the available hydro. data (March 2024)
    def generate_filenames(start_year, start_month):
        filenames = []
        
        # Define the start date and the end date
        start_date = datetime(start_year, start_month, 1)
        end_date = datetime(2024, 3, 31)

        # Loop through each month from the start date to the end date
        current_date = start_date
        while current_date <= end_date:
            year_str = str(current_date.year)
            month_str = str(current_date.month).zfill(2)
            filename = f"{dir}croco_avg_Y{year_str}M{month_str}_surface.nc"
            filenames.append(filename)
            
            # Move to the next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)
        
        return filenames

    # Update month_str within the loop
    month_str  = str(month).zfill(2)
    part_month = f"Y{year}M{month_str}"
    print(f"Month running: {part_month}")
    
    # Generate filenames for the current month
    filenames = generate_filenames(year, month)
    for filename in filenames:
        print(f"Filename running: {filename}")

    # If filenames exist for the current month and year -> run the simulation
    if filenames:
        # PARTICLE RELEASE LOCATIONS AND TIMES
        [X_masked, Y_masked] = np.load(f'INPUT/release_locs_{part_config}_{location}.npy')
        lon_part = X_masked.flatten()
        lat_part = Y_masked.flatten()
        lon_part = lon_part[~np.isnan(lon_part)]  # delete nans
        lat_part = lat_part[~np.isnan(lat_part)]

        # Count how many particles are released:
        npart = len(lon_part)
        print(f"Number of particles released: {npart}")

        # Number of days in the current month
        days = calendar.monthrange(year, month)[1]

        # Time releases array (every 12 hours)
        time_releases = [(day - 1) * 24 * 60 * 60 + hour * 60 * 60 for day in range(1, days + 1) for hour in range(0, 24, seeding_dt)]

        # Create a 1D array of lon, lat, and time for each particle
        lons  = np.tile(lon_part, len(time_releases))
        lats  = np.tile(lat_part, len(time_releases))
        times = np.repeat(time_releases, len(lon_part))

        # Variables, dimensions, indices
        variables  = {'U': 'u', 'V': 'v'}
        dimensions = {'U': {'lon': 'lon_u', 'lat': 'lat_v', 'time': 'time'},
                        'V': {'lon': 'lon_u', 'lat': 'lat_v', 'time': 'time'}}
        indices = {'lon': range(450), 'lat': range(358)}    # In this verison of Parcels indices need
                                                            # to be defned because dimensions of U and V
                                                            # in CROCO do not match
                                                            # NOTE 1: this is needs to be manually changed for 
                                                            # each new model grid 
                                                            # NOTE 2: this is not needed in the newest version of Parcels
                                                            # when using fieldset.from_croco()

        # my particle class has additional variables
        class MyParticle(JITParticle):
            particle_age = Variable('particle_age', dtype=np.float32, initial=0.)
            isNotFrozen = Variable("isNotFrozen", dtype=np.float32, initial=1., to_write=False)

        # fieldset
        fieldset = FieldSet.from_c_grid_dataset(filenames, variables, dimensions, indices=indices)

        # particle set
        pset = ParticleSet(fieldset=fieldset, pclass=MyParticle, lon=lons, lat=lats, time=times)

        # kernels
        def DeleteParticle(particle, fieldset, time):
            if particle.state >= 50:
                particle.delete()

        def FreezeParticle(particle, fieldset, time): # using freezing because deleting of particles is computationally more demanding
            if particle.state >= 50: # If throws an error
                # If throws an error
                particle.isNotFrozen = 0.
                return parcels.StatusCode.StopExecution

        n_particles = pset.size

        print("Running parcels...")
        print(f"Output file: OUT_REGIOCON/{part_config}_{location}_{part_month}.zarr")
        print(f"Days running (days that it's seeding): {days}")
        print(f"Time_releases running this month (every {seeding_dt} hours): {time_releases}")

        # Calculate the runtime from the start of this month until end of March 2024
        end_date = datetime(2024, 3, 31)
        start_date = datetime(year, month, 1)
        runtime = (end_date - start_date).days * 86400  - int(timedelta(days=100).total_seconds())
        print(f"Runtime: {runtime} seconds")

        # Run parcels
        outputfile = pset.ParticleFile(name=f"OUT_REGIOCON/{part_config}_{location}_{part_month}.zarr", outputdt=3600) # outputdt is in seconds

        pset.execute([AdvectionRK4, FreezeParticle], 
                        runtime=timedelta(days=100), # instead of total runtime, run Bonaire for 100 days becuase already checked that way less than 1% of particles remain after 100 days
                        dt=ipart_dt, 
                        output_file=outputfile)
    
        
        print("*** All good ***, 100 days finished, let's go to the next one!")

        # The commented follow-up code does not work for my case, but it works smoothly for example cases:
        # total_runtime = runtime
        # interval_runtime = timedelta(days=10)
        # total_runs = np.ceil(total_runtime/interval_runtime)
        # for i in range(total_runs):
        #     pset.execute([AdvectionRK4,FreezeParticle], runtime=interval_runtime, dt=ipart_dt, output_file=outputfile)
        #     print('Proportion of particles remaining:', pset.isNotFrozen.sum()/n_particles)
        #     if pset.isNotFrozen.sum()/n_particles < 0.005: # that is, less than 5 percent of the particles remain
        #         break

print("FINISHED!")