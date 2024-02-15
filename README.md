<img src="docs/source/_static/icon.png" alt="SCORE logo" width="128"/>

# ***SCORE***: Single-cell Chromatin Organization Representation and Embedding

<div align="center">

A Python package developed by the Jin Lab for combining, benchmarking, and extending methods of embedding, clustering, and visualizing single-cell Hi-C data.

</div>

## Installation

```bash
git clone https://github.com/dylan-plummer/SCORE.git;
pip install .
```

### (Optional) Tensorflow GPU support

Some methods such as Va3DE rely on [GPU accelerated Tensorflow builds](https://www.tensorflow.org/install/pip). Make sure you are using a GPU-build by running

```bash
pip install tensorflow[and-cuda]
```

You can verify that the installation was successful by running

```bash
score --help
```