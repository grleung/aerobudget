These are the scripts associated with the following paper in ACP:
"Aerosol-cloud impacts on aerosol detrainment and rainout in shallow maritime tropical clouds"
Leung et al. (2023)

RAMS source code, a sample RAMS namelist and a sample REVU namelist have also been provided. 
These simulations were run on the NASA Pleiades supercomputer and thus configuration files 
("rams_6.3.03/include.mk") are set up with the paths/modules available there. If you wish to
rerun these simulations, you will have to change the libraries for your specific circumstances.
By default, the RAMS source code assumes regenerated aerosol are sulfate-type in terms of their radiative
properties. To change this, edit the source code in "rams_6.3.03/src/6.3.03/radiate/rad_aero.f90" 
on the indicated lines (79-82). Make sure to recompile RAMS every time this is done (i.e. every 
time running for a different aerosol type.) More information on RAMS can be found at: 
https://vandenheever.atmos.colostate.edu/vdhpage/rams/rams_docs.php

Python analysis scripts should be run before the jupyter notebook for creating the final figures.
The environment can be found under aerobudget.yml to see necessary packages.
Most of these scripts use jug, a package for parallelization of tasks (https://github.com/luispedro/jug);
to run scripts with jug, use "jug <filename.py>". Multiple instances of jug can be run to parallelize tasks.
Scripts beginning with "tobac" also use the tobac package(https://github.com/tobac-project/tobac).

For any further questions about the code or data used in this study, please contact
Gabrielle Leung (gabrielle.leung@colostate.edu). 
