#!/bin/bash

# Output data directory
output_dir=/Users/adminaeverall/Documents/Astro/Data/GaiaDR3RVS_run

# Data extraction parameters
Gres=0.2 # 2 bins per magnitude 0.2
Cres=1 # 1 bins per colour mag interval
healpix=4 # Healpix level

# Count data file
data_file=$output_dir/gaiadr3_rvs_kncounts_Gres${Gres}Cres${Cres}hpx${healpix}.h

# Model fit parameters
jmax=3 #
M=60 # Number of magnitude bins - will end up being 0.5 mag bins
C=4 # Number of colour bins - two mag bins

python QueryDR3.py $output_dir $data_file $Gres $Cres $healpix

python EvaluateDR3RVS.py $output_dir $data_file $M $C $jmax $Gres $Cres $healpix

nside=$(( 2**$healpix ))
results_file=chisquare_rvs_jmax${jmax}_nside${nside}_M${M}_CGR${C}_lm0.3lc300_B2.0_scipy_results.h5
python UseDR3RVS.py $output_dir $results_file
