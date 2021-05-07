#!/bin/bash

for steps in {0..17}
do
     mkdir ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/F4-n_result_0.csv ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/zernike2d_result_0.csv ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/legendre_result_0.csv ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/flux_result_0.csv ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/materials-xz-0.png ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/materials-xy-0.png ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/XY_PLOT_distribution_meshavg_0.gif ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/XZ_PLOT_distribution_meshavg_0.gif ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/PLOT_distribution_interpolated_0.png ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/PLOT_radialflux0.png ../Data_used/OpenMC/step_$steps-CONTROL
     cp ./step_$steps-CONTROL/PLOT_axialflux0.png ../Data_used/OpenMC/step_$steps-CONTROL
done

