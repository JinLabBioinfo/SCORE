#!/usr/bin/env python

## 3DVI
## Author: Ye Zheng
## Contact: yezheng.stat@gmail.com


## Script to remove sequencing depth effect, batch effect and lower dimension project, denoising single-cell Hi-C data.
## March 2021
import contextlib
import io
import sys
import os
import argparse
import pickle
from tqdm import tqdm
import scanpy as sc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA


@contextlib.contextmanager
def nostdout():
    # see: https://stackoverflow.com/questions/2828953/silence-the-stdout-of-a-function-in-python-without-trashing-sys-stdout-and-resto
    save_stdout = sys.stdout
    sys.stdout = io.BytesIO()
    yield
    sys.stdout = save_stdout

def create_band_mat(x: np.ndarray, count: np.ndarray, diag: int, maxChromosomeSize: int) -> np.ndarray:
    bandMat = np.zeros(max(maxChromosomeSize - diag + 1, np.max(x) + 1))
    bandMat[x] = count
    return bandMat[:maxChromosomeSize - diag + 1]


class Process(object):
    def __init__(self, resolution, chromSize=None):
        self._RESOLUTION = resolution
        self._chromSize = chromSize
        self.df = None
        self._lastchrom = None
        self._chormdf = None
    
    def rescale(self, chrA, x, y, counts, resolution = None):
        if resolution:
            self._RESOLUTION = resolution
        
        xR = x // self._RESOLUTION
        yR = y // self._RESOLUTION
        self.df = pd.DataFrame({'chrA': chrA,
                        'x': xR,
                        'y': yR,
                        'counts': counts})
        self.df.loc[:,'diag'] = abs(yR - xR)
        return True
    
    def band(self, chrom, diag, maxBand):
        if self.df is None:
            raise "Run process.rescale(chrA, binA, binY, counts, resolution) first."        
        
        if self._lastchrom is None or (self._lastchrom != chrom):
            self._lastchrom = chrom
            self._chormdf = self.df[self.df.chrA == chrom]            
        
        dat =  self._chormdf[self._chormdf.diag == diag]
        mat = create_band_mat(dat.x.values, dat.counts.values, diag, maxBand)
        return mat
    
    def band_all(self, chromSize, used_chroms = 'whole', used_diags = None):
        if used_diags is None:
            used_diags = [i for i in range(1, 11)]
        if self.df is None:
            raise "Run process.rescale(chrA, binA, binY, counts, resolution) first"
        if chromSize:
            self._chromSize = chromSize
            
        chrom = 'chrA'
        diag_s = 'diag'
                
        cell_band = {}
        for chromosome, chromosome_data in self.df.groupby(chrom):
            if (used_chroms != 'whole' and chromosome not in used_chroms) or chromosome not in self._chromSize:
                continue
            
            bandSize = self._chromSize[chromosome] // self._RESOLUTION + 1
            chromosome_band = {}
            for diag, chromosome_diag in chromosome_data.groupby(diag_s):
                if used_diags != 'whole' and diag not in used_diags:
                    continue
                x = chromosome_diag.x.values
                count = chromosome_diag.counts.values
                chromosome_band[diag] = create_band_mat(x, count, diag, bandSize)
            cell_band[chromosome] = chromosome_band
        return cell_band
    
def read_file(file):
    df = pd.read_csv(file, sep = "\t", header = None, names = ['chrA', 'binA', 'chrB', 'binB', 'counts', 'norm'])
    df.loc[:,'cell'] = file
    return df

def read_file_chrom(file, used_chroms):
    dfTmp = pd.read_csv(file, sep = "\t", header = None, names = ['chrA', 'binA', 'chrB', 'binB', 'counts', 'norm'])
    dfTmp.loc[:,'cell'] = file    
    if used_chroms == 'whole':
        df = dfTmp
    else:
        df = dfTmp[dfTmp.chrA.isin(used_chroms)]

    return df

def read_files(file_list, used_chroms = 'whole', cores = 8):

    df_list = Parallel(n_jobs=cores)(delayed(read_file_chrom)(file, used_chroms) for file in file_list)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df
    
def process_cell(cell_raw: pd.DataFrame, binSizeDict: dict, 
                 resolution: int, cell: str,
                 used_chroms,
                 used_diags):
    process = Process(resolution)
    process.rescale(cell_raw.chrA, cell_raw.binA, cell_raw.binB, cell_raw.counts)    
    cell_band = process.band_all(binSizeDict, used_chroms=used_chroms, used_diags=used_diags)
    return (cell_band, cell)



def get_locuspair(imputeM, chromSelect, bandDist):
    xvars = ['chrA','binA', 'chrB', 'binB', 'count', 'cellID']
    tmp = imputeM.transpose().copy()
    tmp.index.name = 'binA'
    normCount = pd.melt(tmp.reset_index(), id_vars = ['binA'], var_name='cellID', value_name='count')
    normCount.loc[:,'binA'] = normCount.binA.astype(int)
    normCount.loc[:,'binB'] = normCount.binA + bandDist

    normCount.loc[:,'chrA'] = chromSelect
    normCount.loc[:,'chrB'] = chromSelect
    normCount = normCount[xvars]
    
    return normCount


def normalize(bandM, cellInfo, chromSelect, bandDist, nLatent = 64, batchFlag = False, gpuFlag = False, n_epochs=50):
    import scvi
    #print(bandM, chromSelect)
    #bandM = band_chrom_diag[chromSelect][bandDist] #pd.read_csv(args.infile, index_col = 0).round(0)
    #cellSelect = [i for i, val in enumerate(bandM.sum(axis = 1)>0)]
    cellSelect = [i for i, val in enumerate(bandM.sum(axis = 1)>0) if val]
    if len(cellSelect) == 0:
        #print("No cells for this chromosome and distance.")
        normCount = None
        latentDF = pd.DataFrame(np.zeros((len(bandM), nLatent)), index = range(len(bandM)))
        
    else:
        bandDepth = bandM[cellSelect,].sum(axis = 1).mean()
        adata = sc.AnnData(bandM)
    
        if(batchFlag is True):
            adata.obs['batch'] = cellInfo['batch'].values
    
        #sc.pp.filter_cells(adata, min_counts=1)
        try:
            if(batchFlag is True):
                scvi.model.SCVI.setup_anndata(adata, batch_key = 'batch')
            else:
                scvi.model.SCVI.setup_anndata(adata)
            model = scvi.model.SCVI(adata, n_latent = nLatent)
        
            model.train(max_epochs=n_epochs, early_stopping=True, progress_bar_refresh_rate=0)

            if(batchFlag is True):
                imputeTmp = np.zeros((len(cellSelect), bandM.shape[1]))
                for batchName in list(set(cellInfo['batch'].values)):
                    imputeTmp = imputeTmp + model.get_normalized_expression(library_size = bandDepth, transform_batch = batchName)
                imputeM = imputeTmp/len(list(set(cellInfo['batch'].values)))

            else:
                imputeM = model.get_normalized_expression(library_size = bandDepth)
            
            normCount = get_locuspair(imputeM, chromSelect, bandDist)

            latent = model.get_latent_representation()
            latentDF = pd.DataFrame(latent, index = cellSelect)
            latentDF = latentDF.reindex([i for i in range(len(bandM))]).fillna(0)
        except Exception as e:
            print(e)
            normCount = None
            latentDF = pd.DataFrame(np.zeros((len(bandM), nLatent)), index = range(len(bandM)))

    return(latentDF, normCount)


def train_3dvi(bandMax, chromList, resolution, inPath, outdir, cellSummary, genome, batchRemoval, nLatent, gpuFlag, parallelCPU, pcaNum, umapPlot, tsnePlot, batch_removal=False, n_epochs=50):
    import scvi
    ## number of bin per chromosome
    print("Caculate total number of bin per chromosome.")
    used_diags = "whole"
    binSize = pd.read_csv(genome, sep = "\t", header = None)
    binSizeDict = {}
    N = binSize.shape[0]
    for i in range(N):
        chrome = binSize.iloc[i,0]
        size = binSize.iloc[i,1]
        if chrome != 'chrY':
            binSizeDict[chrome] = size
    
    ## cell info file
    print("Prepare cell summary file.")
    if cellSummary is not None:
        cellInfo = pd.read_csv(cellSummary, sep = "\t", header = 0).sort_values(by = 'name')
    else:
        cellName = {'name': os.listdir(inPath)}
        cellInfo = pd.DataFrame(cellName).sort_values(by = 'name')

    cellInfo.index = range(cellInfo.shape[0])
    
    ## read in scHi-C data
    print("Read in scHi-C data.")
    files = list(inPath + '/' + cellInfo.name)
    files.sort()
    
    ## read all the files and sort by cell file name
    resolution = int(resolution)
    coreN = int(parallelCPU)
    used_diags_list = []
    if bandMax == "whole":
        used_diags = "whole"
    else:
        used_diags_list = [i for i in range(1, int(bandMax))]
    
    if chromList == "whole":
        used_chroms = "whole"
    else:
        used_chroms = chromList.split(',')
    
    raws = read_files(files, used_chroms, coreN)
    
    print("Convert interactions into band matrix.")
    if bandMax == "whole":
        raw_cells = Parallel(n_jobs=coreN)(delayed(process_cell)(cell_df, binSizeDict, resolution, cell, used_chroms, used_diags) for cell, cell_df in tqdm(raws.groupby('cell')))
    else:
        raw_cells = Parallel(n_jobs=coreN)(delayed(process_cell)(cell_df, binSizeDict, resolution, cell, used_chroms, used_diags_list) for cell, cell_df in tqdm(raws.groupby('cell')))
    raw_cells.sort(key=lambda x: x[1]) ##x[1] is 'cell', used for sorting
    cells = [cell for _, cell in raw_cells]
    raw_cells = [raw_cell for raw_cell, _ in raw_cells]
    if not os.path.exists(outdir + '/pickle'):
        os.makedirs(outdir + '/pickle', exist_ok=True)
    with open(outdir + '/pickle/raw_cells', 'wb') as f:
        pickle.dump(raw_cells, f)
    # del raws
    # gc.collect()
    
    print("Concat cells into cell x locus-pair matrix.")
    band_chrom_diag = {}
    for chrom, chromSize in binSizeDict.items():
        if used_chroms != "whole" and chrom not in used_chroms:
            continue
        chromSize = chromSize // resolution + 1
        chrom_diag = {}
        for band in range(1, chromSize - 4):
            if used_diags != "whole" and band not in used_diags_list:
                continue
            mat = []
            for fi in range(len(files)):
                if chrom not in raw_cells[fi].keys():
                    tmp = np.zeros(chromSize - band + 1)
                elif band not in raw_cells[fi][chrom]:
                    tmp = np.zeros(chromSize - band + 1)
                else:
                    tmp = raw_cells[fi][chrom][band]
                mat.append(tmp)
            chrom_diag[band] = np.vstack(mat)
        band_chrom_diag[chrom] = chrom_diag
    
    with open(outdir + '/pickle/band_chrom_diag', 'wb') as f:
        pickle.dump(band_chrom_diag, f)
    
    # del raw_cells
    # gc.collect()


    ## 3DVI
    print("3DVI normalization.")
    print(cellInfo)
    bandMiter = [[bandM, chromSelect, bandDist] for chromSelect, band_diags in band_chrom_diag.items() for bandDist, bandM in band_diags.items()]
    nLatent = int(nLatent) #int(args.nLatent)
    batchFlag = batch_removal
    gpuFlag = gpuFlag

    if coreN == 1:
        res = []
        for bandM, chromSelect, bandDist in tqdm(bandMiter):
            res.append(normalize(bandM, cellInfo, chromSelect, bandDist, nLatent, batchFlag, gpuFlag, n_epochs))
    else:
        res = Parallel(n_jobs=coreN,backend='multiprocessing')(delayed(normalize)(bandM, cellInfo, chromSelect, bandDist, nLatent, batchFlag, gpuFlag, n_epochs) for bandM, chromSelect, bandDist in bandMiter)
    # with open(outdir + '/pickle/res', 'wb') as f:
    #     pickle.dump(res, f)
    
    print("Writing out latent embeddings.")
    
    ## concatenate latent embeddings across band matrices
    latentList = [res[i][0] for i in range(len(res))]
    latentM = pd.concat(latentList, axis = 1)
    pca = PCA(n_components = int(pcaNum))
    latentPCA = pca.fit_transform(latentM)
    latentPCA = np.nan_to_num(latentPCA)
    if not os.path.exists(outdir + '/latentEmbeddings'):
        os.mkdir(outdir + '/latentEmbeddings')
    pd.DataFrame(latentPCA).to_csv(outdir + '/latentEmbeddings/norm3DVI_PCA' + str(pcaNum) + '.txt', sep = '\t', header = False, index = False)
    #pd.DataFrame(latentM).to_csv(outdir + '/latentEmbeddings/norm3DVI_latentEmbeddingFull.txt', sep = '\t', header = False, index = False)
    
    
