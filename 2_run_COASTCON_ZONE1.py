"""
Script to run the Parcels simulation for the COASTCON scenario (= Intra-island connectivity)
Release location: zone_1
This script runs in parallel for each year and month, as specified in the .sh file
Needed files: release locaitons of particles, made with 1_release_locations_COASTCON.py
"""

import sys
import numpy as np
from parcels import *

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>; if you see this message, you forgot to add the year and/or month as arguments!")
        sys.exit(1)
    
    year  = int(sys.argv[1])
    month = int(sys.argv[2])
        
    # chosen variables
    part_config   = 'COASTCON' # Parcels scenario name
    zone          = 'zone_1'   # Zone of particle release
    seeding_dt    = 12         # in hours - particles are seeded every seeding_dt hours
    adv_period    = 30         # in days, particles are advected for the period of adv_period days (they are deleted after that)
    ipart_dt      = 5*60       # seconds

    print(f"Start running 2_run_{part_config}_{zone}.py...")
    print("Simulation: SCARIBOS_V8 croco model, releasing particles in the coastal grid every 1/100 degree (approx)")
    print(f"Particles are released every {seeding_dt} hours and are advected for {adv_period} days")
    print('Particle configuration name: '+part_config)
    print("Python kernel used: parcels")
    print("Parcels kernels used: AdvectionRK4, DeleteParticle, UpdateAge, DeleteOldParticles, DeleteFarParticles")

    # Define the base directory - where hydrodynamic data is stored (SCARIBOS)
    dir = '/nethome/berto006/croco/CONFIG/SCARIBOS_V8/CROCO_FILES/surface_currents/'

    # Creates a list of filenames for the current month to be used in Parcels 
    # (2 files per month: current and next month)
    def generate_filenames(month):
        filenames = []
        for i in range(2):
            month_idx = int(month[6:8]) - 1 + i
            year = int(month[1:5])
            if month_idx > 11:
                year += 1
            month_idx %= 12
            month_str = str(month_idx + 1).zfill(2)
            filename = f"{dir}croco_avg_Y{year}M{month_str}_surface.nc"
            filenames.append(filename)
        return filenames

    month_str = str(month).zfill(2)  # Update month_str within the loop
    part_month = f"Y{year}M{month_str}"
    print(f"Month running: {part_month}")
    
    # Generate filenames for the current month
    filenames = generate_filenames(part_month)
    for filename in filenames:
        print(f"Filename running: {filename}")

    # If filenames exist for the current month and year -> run the simulation
    if filenames:
        # PARTICLE RELEASE LOCATIONS AND TIMES
        # Load parcels release locations from file (made with 1_release_locations_COASTCON.py)
        [X_masked, Y_masked] = np.load(f'INPUT/release_locs_{part_config}_{zone}.npy')
        lon_part = X_masked.flatten()
        lat_part = Y_masked.flatten()
        lon_part = lon_part[~np.isnan(lon_part)]  # delete nans
        lat_part = lat_part[~np.isnan(lat_part)]

        # Count how many particles are released:
        npart = len(lon_part)
        print(f"Number of particles released: {npart}")

        # Number of days
        days = 31 if month in [1, 3, 5, 7, 8, 10] else 30 if month != 2 else 28

        # Time releases array (every 12 hours)
        time_releases = [(day - 1) * 24 * 60 * 60 + hour * 60 * 60 for day in range(1, days + 1) for hour in range(0, 24, seeding_dt)]

        # Create a 1D array of lon, lat, and time for each particle
        lons  = np.tile(lon_part, len(time_releases))
        lats  = np.tile(lat_part, len(time_releases))
        times = np.repeat(time_releases, len(lon_part))

        # variables, dimension, indices
        variables  = {'U': 'u', 'V': 'v'}
        dimensions = {'U': {'lon': 'lon_u', 'lat': 'lat_v', 'time': 'time'},
                      'V': {'lon': 'lon_u', 'lat': 'lat_v', 'time': 'time'}}
        indices    = {'lon': range(450), 'lat': range(358)} # In this verison of Parcels indices need
                                                            # to be defned because dimensions of U and V
                                                            # in CROCO do not match
                                                            # NOTE 1: this is needs to be manually changed for 
                                                            # each new model grid 
                                                            # NOTE 2: this is not needed in the newest version of Parcels
                                                            # when using fieldset.from_croco()
        
        # my particle class: adding age as a variable
        class MyParticle(JITParticle):
            particle_age = Variable('particle_age', dtype=np.float32, initial=0.)

        # fieldset
        fieldset = FieldSet.from_c_grid_dataset(filenames, variables, dimensions, indices=indices)

        # particleset
        pset = ParticleSet(fieldset=fieldset, pclass=MyParticle, lon=lons, lat=lats, time=times)

        # kernels
        def DeleteParticle(particle, fieldset, time):
            if particle.state >= 50:
                particle.delete()

        def UpdateAge(particle, fieldset, time):
            if particle.time > 0:
                particle.particle_age += particle.dt

        def DeleteOldParticles(particle, fieldset, time):
            if particle.particle_age > adv_period * 86400:  # Check if age exceeds 30 days (in seconds)
                particle.delete()

        def DeleteFarParticles(particle, fieldset, time):
            if particle.lon < -69.49 or particle.lon > -68.49 or particle.lat < 11.67 or particle.lat > 12.67:
                particle.delete()

        print("Running parcels...")
        print(f"Output file: OUT_COASTCON/{part_config}_{zone}_{part_month}.zarr")
        print(f"Days running (days that it's seeding): {days}")
        print(f"Time_releases running this month (every {seeding_dt} hours): {time_releases}")
        print("Output dt = 60min")

        # output file
        outputfile = pset.ParticleFile(name=f"OUT_COASTCON/{part_config}_{zone}_{part_month}.zarr", outputdt=3600) # outputdt: 3600 = 1h

        # execute
        pset.execute([AdvectionRK4, DeleteParticle, UpdateAge, DeleteOldParticles, DeleteFarParticles],
                    runtime=86400 * days*2, 
                    dt=ipart_dt, 
                    output_file=outputfile)
        
        print("*** All good ****, month finished, let's go to the next one!")

print("FINISHED!")
