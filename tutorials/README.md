### Method usage details

#### [scHiCluster](https://doi.org/10.1073/pnas.1901423116) (`scHiCluster`)

* `--random_walk_iter <N>` to set number of random walk iterations (default=1)

* `--random_walk_ratio <r>` analogous to restart probability (default=1.0)

* `--no_preprocessing` will skip the default VC_SQRT_norm + Convolution + Random Walk preprocessing

#### [fastHiCRep+MDS](https://doi.org/10.1093/bioinformatics/bty285) (`fastHiCRep`) / [InnerProduct+MDS](https://doi.org/10.1371/journal.pcbi.1008978) (`InnerProduct`)


* `--n_strata <N>` to control number of distal strata (default=32) 

* `--strata_offset <N>` to control number of strata to ignore (default=0)

* `--viz_innerproduct` to generate an inner product heatmap in the results directory


#### [scHi-C Topic Modeling](https://doi.org/10.1371/journal.pcbi.1008173) (`cisTopic`)

* `--n_strata <N>` to control number of distal strata (default=32) 

* `--cistopic_minc <N>` minimum number of topics to consider in search (default=8)

* `--cistopic_maxc <N>` maximum number of topics to consider in search (default=64)


#### [SnapATAC2](https://doi.org/10.1038/s41592-023-02139-9) (`SnapATAC`)

* `--n_strata <N>` to control number of distal strata (default=32) 

* `--snapatac_features <N>` to set number of variable features to use (default=500000)

* `-snapatac_upper_q <q>` to set upper quantile for filtering (default=0.005)

* `--snapatac_lower_q <q>` to set lower quantile for filtering (default=0.005)

* `--snapatac_max_iter <N>` to set maximum number of feature selection iterations (default=1)
 

#### [scGAD](https://doi.org/10.1093/bioinformatics/btac372) (`scGAD`) (requires [additional R dependencies](https://sshen82.github.io/BandNorm/articles/BandNorm-tutorial.html))

* `--depth_norm` to apply depth normalization (default=FALSE)


#### [Insulation Scores](https://doi.org/10.1038/nature14450) (`Insulation`)

* `--n_strata <N>` to control number of distal strata (default=32) 

#### [DeTOKI](https://doi.org/10.1186/s13059-021-02435-7) (`deTOKI`)

* `--n_strata <N>` to control number of distal strata (default=32) 


#### [scVI-3D](https://doi.org/10.1186/s13059-022-02774-z) (`3DVI`)

* `--n_strata <N>` to control number of distal strata (default=32) 

* `--batch_correct` to apply batch correction within the scVI model (default=FALSE)


* [Higashi](https://doi.org/10.1038/s41587-021-01034-y) (`Higashi`)

* `--higashi_epochs <N>` to set number of epochs (default=60)

* `--higashi_n_strata <N>` to set number of strata (default=None) *EXPERIMENTAL* sets interactions to zero to simulate distance truncation, Higashi considers all interactions by default

* `--higashi_strata_offset <N>` to set number of strata to ignore (default=0) *EXPERIMENTAL* sets interactions to zero to simulate strata offset

* `--higashi_dryrun` just build the hypergraph and initial SVD embedding but do not train (useful to checking if you have set things up correctly)


* [Fast Higashi](https://doi.org/10.1016/j.cels.2022.09.004) (`fast_higashi`/`fast-higashi`)

* `--n_strata <N>` to control number of distal strata (default=100) 

* `--fast_higashi_tol <N>` to set convergence tolerance (default=2e-5)


#### [Va3DE](https://github.com/JinLabBioinfo/Va3DE) (`VaDE`/`Va3DE`)

1. `--binarize` to model binary occurrence of interactions (Bernoulli decoder) 

2. `--counts` to model counts of interactions (Poisson decoder)

* `--n_strata <N>` to control number of distal strata (default=32) 

* `--beta <v>` set the weight of the KL-divergence term (default=1.0)

* `--n_clusters <N>` set the number of prior assumed clusters (default=10 for small datasets, 30 for large datasets)

* `--start_filters <N>` set the number of filters in the first layer of the encoder (default=4)

* `--start_filter_size <N>` set the size of the first filters (default=5)

* `--stride <N>` set the stride of each layer to reduce dimensionality (default=2)

* `--stride_y` apply stride to both axes (both across genome length and across distances) (default=FALSE)

* `--n_epochs <N>` set the number of epochs (default=1000)

* `--batch_size <N>` set the batch size (default=64)

* `--lr <v>` set the learning rate (default=1e-4)

* `--weight_decay <v>` set the weight decay for AdamW (default=1e-4)

* `--load_vade_from <N>` load a pre-trained embedding from a given epoch (default=None)