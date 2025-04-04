import os
import sys, getopt, time
import multiprocessing
import warnings

from sklearn import decomposition
import pandas as pd
import numpy as np
import math

warnings.filterwarnings('ignore')

def usage():
    print("""Usage: path/to/python3 path/to/TOKI.py <Hi-c matrix file> options
    Options:
     -b <kbp resolution of matrix> (default=40)
     -o <output dir> (default=./TAD)
     -s <TAD mean kbp size min,max> (default=600,1000)
     -l <kbp size of split window> (default=8000)
     -p <number of cores to use> (default=1)""")

## Run NMF in several times with random initialisation and output consensus matrix
def corate(A,n,time):
    S=np.zeros([np.shape(A)[0],np.shape(A)[1]])
    for i in range(time):
        K=np.zeros([np.shape(A)[0],np.shape(A)[1]])
        estimator=decomposition.NMF(n_components=n, init='random',random_state=i)
        estimator.fit(A)
        B=estimator.transform(A)
        index=B.argmax(axis=1)
        for j in range(n):
            for s in np.where(index==j)[0]:
                for t in np.where(index==j)[0]:
                    K[s,t]=1
        S=S+K
    return S.astype(np.float64)/time

## Calculate clustering rate of each bin
def IS(R, length):
    bias=np.zeros([np.shape(R)[0]])
    for i in range(1,np.shape(R)[0]-1):
        bias[i]=np.mean(R[max(0,i-length):(i+1),i:min(i+length+1,np.shape(R)[0])])
    return bias

## Find bins with local minimal clustering rate and global comparative low clustering rate （these bins are detected TAD boundaries)
def zero(R,t, delta, length):
    bias=IS(R, length)
    delta_list=[]
    for i in range(delta,len(bias)-delta):
        delta_list.append(-sum(bias[(i-delta):i])+sum(bias[i:(i+delta)]))
    zero=[0]
    for i in range(len(delta_list)-1):
        if delta_list[i]<0 and delta_list[i+1]>=0:
            #zero.append(i+5-np.argmin([bias[i+x] for x in range(5,0,-1)]))
            zero.append(i+delta)
    zero.append(len(bias))
    zero=sorted(np.unique(zero))
    enrich=[0]
    strength=[]
    for j in range(1,len(zero)-1):
        strength.append(max(max(bias[zero[j-1]:(zero[j]+1)])-bias[zero[j]],max(bias[(zero[j]-1):zero[j+1]])-bias[zero[j]]))
    strength=np.array(strength)
    index=sorted(list(set(np.argsort(-strength)[:t])|set(np.where(strength>0.3)[0])))
    for k in index:
        enrich.append(zero[k+1])
    enrich.append(len(bias))
    return np.array(enrich)

## Define silhouette coefficient of consensus matrix and detected TAD boundaries
def silhou(R,pos):
    n=np.shape(R)[0]
    silhou=0
    for i in range(len(pos)-1):
        for j in range(pos[i],pos[i+1]):
            a=np.sum((1-R)[j,pos[i]:pos[i+1]])
            b=np.sum((1-R)[i,:])-a
            silhou+=(-a/(pos[i+1]-pos[i])+b/(n+pos[i]-pos[i+1]))/max((a/(pos[i+1]-pos[i]),b/(n+pos[i]-pos[i+1])))
    return silhou/n

## Find the best n_components by comparing silhouette coefficient
def bestco(F, resolution, size, delta, length):
    x=-1
    R0=0
    n1=0
    for n in range(max(int(np.shape(F)[0]*resolution/size[1]),1),int(np.shape(F)[0]*resolution/size[0])+1):
        R1=corate(F,n,5)
        x1=silhou(R1,zero(R1,n-1, delta, length))
        if x1>=x:
            R0=R1
            x=x1
            n1=n
    return R0,n1

## Split huge contact matrix to windows and detected TAD boundaries in each window
def part_zero(F, window, core, resolution, size, delta, length):
    pos=[]
    try:
        n=np.shape(F)[0]
    except IndexError:  # empty contact matrix (e.g chrM)
        return []
    global task
    def task(i):
        P=F[max(0,window//2*i-window//4):min(n,window//2*i+window-window//4),max(0,window//2*i-window//4):min(n,window//2*i+window-window//4)]
        #if np.sum(P)<100:
        #    return []
        R,t=bestco(P, resolution, size, delta, length)
        if t==0:
            return []
        p=zero(R,t-1, delta, length)
        pos=[]
        for j in p:
            if j in range(window//2*i-max(0,window//2*i-window//4),window//2*(i+1)-max(0,window//2*i-window//4)):
                pos.append(j+max(0,window//2*i-window//4))
        return pos
    pool = multiprocessing.Pool(processes=core)
    res_list=[]
    for i in range(math.ceil(2*n/window)):
        result = pool.apply_async(task, args=(i,))
        res_list.append(result)
    pool.close()
    pool.join()
    for i in range(math.ceil(2*n/window)):
        result=res_list[i]
        try:
            pos = np.append(pos, result.get())
        except Exception as e:
            pass
    return np.int32(pos)


def run_detoki(input_file, output_file, resolution, size, core, split, delta_scale=1.0):

    ## Limit each NMF to one core running
    c='1'

    ## Import required modules


    t=time.time()

    length = max(1, int(4000 / resolution * delta_scale))
    #length=400//resolution
    #delta=int(math.ceil(100/resolution))
    delta = int(2000 / resolution / delta_scale)
    #window=split//resolution
    window= max(1, int(8000 / resolution * delta_scale))

    #size = [600, 4000]

    ## Input the contact matrix and output the detected TAD boundaries
    F=np.loadtxt(input_file)
    #F=np.array(pd.read_csv(input_file,delimiter='\t',index_col=0))
    # try:
    l=part_zero(F, window, core, resolution, size, delta, length)
    np.savetxt(output_file,l,fmt='%s')
