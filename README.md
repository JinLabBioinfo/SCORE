<img src="docs/source/_static/icon.png" alt="SCORE logo" width="128"/>

# ***SCORE***: Single-cell Chromatin Organization Representation and Embedding

<div align="center">

[![Build status](https://github.com/dylan-plummer/SCORE/actions/workflows/build.yml/badge.svg)](https://github.com/dylan-plummer/SCORE/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/SCORE.svg)](https://pypi.org/project/SCORE/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/dylan-plummer/SCORE/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/dylan-plummer/SCORE/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/dylan-plummer/SCORE/releases)
[![License](https://img.shields.io/github/license/dylan-plummer/SCORE)](https://github.com/dylan-plummer/SCORE/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)

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

## 🛡 License

[![License](https://img.shields.io/github/license/dylan-plummer/SCORE)](https://github.com/dylan-plummer/SCORE/blob/master/LICENSE)

This project is licensed under the terms of the `MIT` license. See [LICENSE](https://github.com/dylan-plummer/SCORE/blob/master/LICENSE) for more details.
