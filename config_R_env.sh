conda install -c conda-forge r-base  # install R (>=4.0)
# for some reason certain R dependencies cannot be installed using install.packages('.')
# run this before trying to install BandNorm and/or cisTopic
conda install -c conda-forge r-gert
conda install -c conda-forge r-rgeos
conda install -c conda-forge r-devtools
# conda install r-harmony
# conda install -c conda-forge r-arrow
# conda install -c conda-forge signac
#conda install bioconductor-genomicinteractions  # required for original cisTopic implementation
Rscript install_R_packages.R