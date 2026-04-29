import math

###########################################
###########################################

# Geometry - box
# DO NOT FORGET TO CHANGE THE .GEO AND .MSH FILES ACCORDINGLY

box_depth = 2.9e6 
box_width = 8.0e6
z_LM = 660.0e3

###########################################

# Geometry - plates

R_bending = 250.0e3
alpha = math.pi/14.3

X_start_SP = 0.0e3
X_end_SP = 4.0e6
X_end_OP = 8.0e6


Age_SP = Age_SP_in_yr*365.0*86400.0
Age_OP = Age_OP_in_yr*365.0*86400.0
# Initial ages of the plates at the trench, in seconds

u_SP = (X_end_SP-X_start_SP)/Age_SP
u_OP = (X_end_OP-X_end_SP)/Age_OP
# plate velocities for initial thermal structure

delta_x = 1.0e3

###########################################

# Geometry - weak layer

weak_layer_thickness = 8.0e3
z_switch_off = 180.0e3
thickness_transition_switch_off_weak_layer = 20.0e3
# on which thickness the transition from "weak" to "normal" material is made

yield_strength_surface_weak = 2.0e6
friction_coefficient_weak = 0.02
yield_strength_max_weak = 1.0e10

max_viscosity_weak_layer = 1.0e20
percentage = 0.01

###########################################
###########################################

# Rheology

mu_max = 1.0e25
mu_min = 1.0e18

#######

# Byerlee

yield_strength_surface = 2.0e6
yield_strength_max = 1.0e10
friction_coefficient = 0.2

#######

# Diffusion creep

Ediff_UM=300.0e3
Ediff_LM=200.0e3
        
Vdiff_UM=4.0e-6
Vdiff_LM=1.5e-6
        
Adiff_UM=3.0e-11
Adiff_LM=6.0e-17

thickness_viscosity_jump = 20.0e3

#######

# Dislocationn creep

n=3.5

Adisl_UM=5.0e-16
Edisl_UM=540.0e3
Vdisl_UM=12.0e-6

Adisl_LM=1.0e-42
Edisl_LM=300.0e3
Vdisl_LM=2.0e-6

#######

# Peierls

Ap_UM=1.0e-150;
Ep_UM=540.0e3;
Vp_UM = 10.0e-6;

Ap_LM=1.0e-300;
Ep_LM=300.0e3;
Vp_LM = 2.0e-6;

n_peierls = 20.0;


### Weak asthenospheric layer inputs
T_LAB = 1373.; 		   # Temperature defining LAB 
zmin_weakasth    =  0e3;   # Depth of top of weakened materials
zmax_weakasth    =  220e3; # Depth of base of weak asthenosphere
coeff_mu_weakaSP =  0.5;   # Ratio of viscosity weak asthenosphere/asthenosphere below SP
coeff_mu_weakaOP =  0.5;   # Ratio of viscosity weak asthenosphere/asthenosphere below OP


###########################################
###########################################

# mantle geotherm

UM_gradient = 0.5
LM_gradient = 0.3

###########################################

# physical constants

gravity = 9.8
gas_constant = 8.3145
T_surface = 273.0
T_mantle = 1573.0

###########################################

# other plate parameters

density = 3300.0
thermal_diffusivity = 1.0e-6

###########################################
###########################################




