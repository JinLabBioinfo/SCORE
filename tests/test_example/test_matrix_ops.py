import os
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from score.__main__ import app, version_callback
from score.utils.matrix_ops import convolution, random_walk, OE_norm, KR_norm, VC_SQRT_norm, \
                                    network_enhance, network_centrality, graph_google, \
                                    graph_modularity, graph_resource_allocation, graph_jaccard, graph_adamic_adar, \
                                    graph_preferential_attachment, graph_mst, graph_coloring, graph_min_edge_cut, \
                                    quantile_cutoff

def random_mat():
    mat = np.random.poisson(0.1, (100, 100))
    mat = mat + mat.T
    return mat

def test_conv():
    mat = random_mat()
    mat = convolution(mat)
    assert True

def test_random_walk():
    mat = random_mat()
    mat = random_walk(mat)
    assert True

def test_dist_norm():
    mat = random_mat()
    mat = OE_norm(mat)
    assert True

def test_kr_norm():
    mat = random_mat()
    mat = KR_norm(mat)
    assert True

def test_vc_sqrt_norm():
    mat = random_mat()
    mat = VC_SQRT_norm(mat)
    assert True

def test_network_enhance():
    mat = random_mat()
    mat = network_enhance(mat)
    assert True

def test_network_centrality():
    mat = random_mat()
    mat = network_centrality(mat)
    assert True

def test_graph_google():
    mat = random_mat()
    mat = graph_google(mat)
    assert True

def test_graph_modularity():
    mat = random_mat()
    mat = graph_modularity(mat)
    assert True

def test_graph_resource_allocation():
    mat = random_mat()
    mat = graph_resource_allocation(mat)
    assert True

def test_graph_jaccard():
    mat = random_mat()
    mat = graph_jaccard(mat)
    assert True

def test_graph_adamic_adar():
    mat = random_mat()
    mat = graph_adamic_adar(mat)
    assert True

def test_graph_preferential_attachment():
    mat = random_mat()
    mat = graph_preferential_attachment(mat)
    assert True

def test_graph_mst():
    mat = random_mat()
    mat = graph_mst(mat)
    assert True

def test_graph_coloring():
    mat = random_mat()
    mat = graph_coloring(mat)
    assert True

def test_graph_min_edge_cut():
    mat = random_mat()
    mat = graph_min_edge_cut(mat)
    assert True

def test_quantile_cutoff():
    mat = random_mat()
    mat = quantile_cutoff(mat)
    assert True