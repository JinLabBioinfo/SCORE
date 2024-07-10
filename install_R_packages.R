## Default repo
local({r <- getOption("repos")
       r["CRAN"] <- "http://cran.r-project.org" 
       options(repos=r)
})

install.packages(c('ggplot2', 'dplyr', 'data.table', 'Rtsne', 'umap', 'lgr', 'rlist'))

if (!requireNamespace("devtools", quietly=TRUE))
    install.packages("devtools")

devtools::install_github('sshen82/BandNorm', build_vignettes = FALSE)
library(BandNorm)

#devtools::install_github("aertslab/cisTopic")