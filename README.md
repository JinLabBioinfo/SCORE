<img src="docs/source/_static/icon.png" alt="SCORE logo" width="128"/>

# ***SCORE***: Single-cell Chromatin Organization Representation and Embedding



A Python package developed by the Jin Lab for combining, benchmarking, and extending methods of embedding, clustering, and visualizing single-cell Hi-C data.

## Installation

```bash
git clone https://github.com/JinLabBioinfo/SCORE.git;
cd SCORE;
pip install .
```

Installation should only take a few minutes.

### (Optional) Tensorflow and PyTorch GPU support

Some methods such as Va3DE rely on [GPU accelerated Tensorflow builds](https://www.tensorflow.org/install/pip). Make sure you are using a GPU-build by running

```bash
pip install tensorflow[and-cuda]
```

We also provide Va3DE as a standalone package which can be installed here: https://github.com/JinLabBioinfo/Va3DE

Other methods such as Higashi rely on [GPU accelerated builds of PyTorch](https://pytorch.org/get-started/locally/).

### SCORE Usage

You can verify that the installation was successful by running

```bash
score --help
```

### Tutorials

We provide some tutorials to help you get started:

* [Introduction (Creating .scool files and generating your first embeddings)](https://github.com/JinLabBioinfo/SCORE/blob/41a0ed371ba8ef00bff3a13c9d871bae116a1d1b/tutorials/intro.ipynb)

### Supported Embedding Methods

The following embedding methods can be run using the `--embedding_algs` argument (not case sensitive):

* [scHiCluster](https://doi.org/10.1073/pnas.1901423116) (`scHiCluster`)
* [fastHiCRep+MDS](https://doi.org/10.1093/bioinformatics/bty285) (`fastHiCRep`)
* [InnerProduct+MDS](https://doi.org/10.1371/journal.pcbi.1008978) (`InnerProduct`)
* [scHi-C Topic Modeling](https://doi.org/10.1371/journal.pcbi.1008173) (`cisTopic`)
* [SnapATAC2](https://doi.org/10.1038/s41592-023-02139-9) (`SnapATAC`)
* [scGAD](https://doi.org/10.1093/bioinformatics/btac372) (`scGAD`) (requires [additional R dependencies](https://sshen82.github.io/BandNorm/articles/BandNorm-tutorial.html))
* [Insulation Scores](https://doi.org/10.1038/nature14450) (`Insulation`)
* [DeTOKI](https://doi.org/10.1186/s13059-021-02435-7) (`deTOKI`)
* [scVI-3D](https://doi.org/10.1186/s13059-022-02774-z) (`3DVI`)
* [Higashi](https://doi.org/10.1038/s41587-021-01034-y) (`Higashi`)
* [Fast Higashi](https://doi.org/10.1016/j.cels.2022.09.004) (`fast_higashi`/`fast-higashi`)
* [Va3DE](https://github.com/JinLabBioinfo/Va3DE) (`VaDE`/`Va3DE`)

We also provide additional baseline methods for benchmarking:

* `1D_PCA` (sum all interactions at each bin, embed 1D counts with PCA)
* `2D_PCA` (extract band of interactions, embed with PCA)
* `scVI` (sum all interactions at each bin, train scVI model)
* `scVI_2D` (extract band of interactions, train scVI model)

### Basic CLI usage

We provide a small example dataset in the `examples/data` directory. To run `SCORE` you simple need to provide an input `.scool` file and a metadata reference file. You can specify the embedding tool(s) you wish to test using the `--embedding_algs` argument

```bash
score embed --dset oocyte_zygote \  # name for saving results
            --scool oocyte_zygote_mm10_1M.scool \  # path to scool file
            --reference oocyte_zygote_ref \  # metadata reference
            --embedding_algs InnerProduct \  # embedding method name
            --n_strata 20 \
```

This will create a new `results` directory (or a directory specified by `--out`) where results are stored under the name specified by `--dset`. Visualizations are generated for celltypes and other metadata provided, and if multiple celltype labels are provided, clustering metrics will be computed and stored as well. Additional analysis and visualization can be easily performed with the `anndata_obj.h5ad` Scanpy object which is saved with each run. Most baseline methods on this small dataset should only take a few minutes to run.

We also provide the datasets analyzed in our benchmark publication at various resolutions which can be downloaded from the following to reproduce our results:

```bash
wget hiview10.gene.cwru.edu/~xww/scHi-C_data.tar.gz
```

For example, to reproduce the short-range complex tissue analysis, we can run:

```bash
score embed --dset pfc \  # name for saving results
            --scool pfc_200kb.scool \  # path to scool file
            --reference pfc_ref \  # metadata reference
            --embedding_algs InnerProduct \  # embedding method name
            --n_strata 10 \  # 0-2Mb
            --min_depth 50000  # filter low depth cells
```

![](assets/images/embedding_shortrange.jpg)

```bash
score embed --dset pfc \  # name for saving results
            --scool pfc_200kb.scool \  # path to scool file
            --reference pfc_ref \  # metadata reference
            --embedding_algs InnerProduct \  # embedding method name
            --strata_offset 10 \  # ignore first 10 strata (i.e 0-2Mb)
            --n_strata 100 \
            --min_depth 50000
```

![](assets/images/embedding_longrange.jpg)


Including multiple embedding methods and executing multiple runs using `--n_runs` will produce a local benchmark on the dataset provided:

```bash
score embed --dset embryo \  # name for saving results
            --scool embryo_500kb.scool \  # path to scool file
            --reference embryo_ref \  # metadata reference
            --ignore_filter \  # keep all cells
            --embedding_algs 1d_pca InnerProduct scHiCluster \
            --n_runs 10
```