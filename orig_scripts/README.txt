#####-----   SUMMARY   -----#####
This repository contains data to reproduce the results of the manuscript :
"The effect of a weak asthenospheric layer on surface kinematics, subduction dynamics and slab morphology in the lower mantle" 
by Cerpa, N. G., Sigloch, K., Garel, F., Heuret, A., Davies, D. R., Mihalynuk, M., Journal of Geophysical Research : Solid Earth.

The data includes the output files of the reference simulations of the standard and the wal cases, as well as the fluidity input files to reproduce all simulations.


#####----- INSTRUCTIONS -----#####  
* Under folder OUTPUTS, the two folders named STD_RefModel and WAL_RefModel contain output files of the reference simulation for the standard case and that for the WAL case (alpha=0.5), respectively. You can use ParaView available at https://www.paraview.org/download/, to visualise the outputs.

* All other simulations can be reproduced by using the input file for fluidity (subductionLM-multi2WAL-hmin800m_v1.flml), the parameter file (constants_weakasth2.py) and the mesh file (meshLM_8000km.msh). In constants_weakasth2.py, change Age_SP_in_yr and Age_OP_in_yr to the age of both plates, and change the parameters in the section "Weak asthenospheric layer inputs" to modify the WAL parameters (weakening factor, depth of th bottom, ...). To run the simulations you must download and compile Fluidity, available at https://github.com/FluidityProject/fluidity. 

