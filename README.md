# SCARIBOS_ConnectivityCuracao

Repository for Vesna's project on flow patterns, hotspots, and connectivity around Curaçao (Southern Caribbean). Preprint: Bertoncelj, V., Mienis, F., Stocchi, P., and van Sebille, E.: Flow patterns, hotspots and connectivity of land-derived substances at the sea surface of Curaçao in the Southern Caribbean, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-3112, 2024.

This project contains the following folders:

## Project Structure:

- **[SCARIBOS_model](./SCARIBOS_model/)**: SCARIBOS model scripts, configurations, and analysis.
    - **[SCARIBOS_run_files](./SCARIBOS_model/SCARIBOS_run_files/)**: Configuration files required to run the SCARIBOS model.
    - **[SCARIBOS_analysis](./SCARIBOS_model/SCARIBOS_analysis/)**: Scripts to analyse SCARIBOS model outputs: validation and plotting of average currents.
- **[parcels_analysis](./parcels_analysis/)**: Contains the scripts related to particle tracking and trajectory analysis using OceanParcels.


## SCARIBOS model:

SCARIBOS: hydrodynamic model of the South CARIBbean Ocean System, made with CROCO community model. 

SCARIBOS is run for the period from December 2019 to March 2024. Period from December 2019 to March 2020 is discarded from further analysis, to account for the spin-up time. The model is executed in monthly intervals (in CROCO referred to as: interannual runs), and the output is saved at the end of each month.

### Information about the model: 
- 3D hydrodynamic model
- CROCO community model, version 1.3.1: Auclair, F., Benshila, R., Bordois, L., Boutet, M., Brémond, M., Caillaud, M., Cambon, G., Capet, X., Debreu, L., Ducousso, N., Dufois, F., Dumas, F., Ethé, C., Gula, J., Hourdin, C., Illig, S., Jullien, S., Le Corre, M., Le Gac, S., Le Gentil, S., Lemarié, F., Marchesiello, P., Mazoyer, C., Morvan, G., Nguyen, C., Penven, P., Person, R., Pianezze, J., Pous, S., Renault, L., Roblou, L., Sepulveda, A., and Theetten, S.: Coastal and Regional Ocean COmmunity model (1.3.1), Zenodo [code], https://doi.org/10.5281/zenodo.7415055, 2022.
- Horizontal resolution: 1/100°
- Vertical resolution: 50 sigma-depth layers
- Horizontal extent: Longitudes: 70.5°W to 66.0°W; Latitudes: 10.0°N to 13.5°N
- Bathymetry input: global product GEBCO (version 2023; GEBCO Compilation Group, 2023) and bathymetry around Curaçao obtained using multibeam sonar during RV Pelagia expedition 64PE500
- Oceanographic initial and boundary conditions: GLORYS12V1 (Lellouche et al., 2021) - created using CROCO TOOLS product (V1.3.1)
- Atmospheric forcing: ERA-5 global atmosphere reanalysis (Hersbach et al., 2020)
- River runoff: four rivers Tocuyo, Yaracuy, Tuy and Grande - created using CROCO TOOLS product (V1.3.1)

### Scripts used for SCARIBOS model simulation and analysis:

**[SCARIBOS_run_files](./SCARIBOS_model/SCARIBOS_run_files/)**:
- ```cppdefs.h```: defining regional configuration
- ```croco_grd.nc```: grid file where model domain and bathymetry are stored
- ```croco_inter.in```: .in file used in interannual simulations
- ```croco_runoff.nc```: data on runoff of the four rivers in the domain
- ```crocotools_param.m```: parameters used in CROCO TOOLS product (V1.3.1)
- ```jobcomp```: compilation file, specific to run on IMAU Lorenz cluster
- ```param.h```: list of model parameters
- ```run_croco_inter.bash```: script that defines and runs interannual simulations (creates new files for each month)
- ```submit_inter.sh```: script that sends SLURM job to IMAU Lorenz cluster to run in parallel

**[SCARIBOS_analysis](./SCARIBOS_model/SCARIBOS_analysis/)**:
- ```1_PE529_extract_surface_currents.py```: script that extract the surface currents from ADCP data, obtained with VMADCP during RV _Pelagia_ 64PE529 expedition
- ```1_plot_map_of_SouthernCaribbean.py```: script to re-create **Figure 1** of the manuscript
- ```1_validate...```: scripts to validate the model with observations (ADCP from RV _Pelagia_ and tidal gauge) and re-create **Figure 3** and **Figure 4**
- ```2_avg_meri_ALLYEARS.py``` and ```2_avg_meri_MONTHLY.py```: scripts to calcualte average meridional currents (all years together and monthly variations) and re-create **Figure 7**
- ```2_avg_surface_ALLYEARS.py``` and ```2_avg_surface_MONTHLY.py```: scripts to calcualte average surface currents (all years together and monthly variations) and re-create **Figure 6**
- ```2_merge_plots_for_paper.py```: script that merges plots of monthly and yearly average currents (either meridioanl or surface) and create final figures **Figure 6** and **Figure 7**
- ```3_validation_...```: scripts for validation of SCARIBOS model (with GlobCUrrent, MultiObs and GLORYS), to re-create **Figure 2** and supplimentary figures **Figure S2**, **Figure S3**, **Figure S4**, **Figure S5** and **Figure S6**.
- ```4_checking_EKE_for_spinup_time.py```: script to calculate EKE (for spin-up time duration) and to re-create supplimantary **Figure S1**


## Lagrangian particle tracking using Parcels: simulations and diagnostics

In this project the Parcels version 3.0.3 is used. The analysis is performed using the SCARIBOS hydrodynamic model output. Particle tracking simulations are used to model the movement of passive particles, representing nutrients and pollutants, in the uppermost layer of the SCARIBOS model. This represents positively buoyant substances that move with the surface flow conditions.

Three scenarios are performed: 
- **Scenario 1**: _Hotspots around Curaçao_ (in scripts referring to: **HOTSPOTS**) - creating maps of probability density functions (PDFs) to identify hotspots around Curaçao
- **Scenario 2**: _Intra-island connectivity_ (in scripts referring to: **COASTCON**) - connectiviy between the 8 coastal zones of Curaçao
- **Scenario 3**: _Coastal connectivity_ (in scripts referring to: **REGIOCON**) - connectivity of neighbouring islands (Aruba, Bonaire and Venezuelan islands) and a part of Venezuelan continental coastline with Curaçao

The structure of the scripts, found in **[parcels_analysis](./parcels_analysis/)**, is as follows: 
- ```0_cut_surface_of_SCARIBOS.py```: this script is needed to cut only the uppermost layer of SCARIBOS, required for Parcels simulations
- ```1_...```: these are pre-processign scripts, used to create release locations of particles, polygon around Curaçao and plot the scenarios
- ```2_...```: these scripts are used to run Parcels for each scenario
- ```3_...```: these scripts are used to perform Lagrangian diagnostics (calculating PDFs and connectivity)
- ```4_...```: these scripts are used to plot results (see below)
- ```submit_...```: these are 6 example scripts that are used to submit SLURM jobs to IMAU Lorenz cluster (usually to run in parallel, but not necessary)

Scripts to re-create figures in the manuscript: 
- ```1_plot_scenario_maps.py```: **Figure 5**
- ```4_plot_HOTSPOTS.py```: **Figure 8**
- ```4_plot_COASTCON.py```: **Figure 9**
- ```4_plot_REGIOCON.py```: **Figure 10**

