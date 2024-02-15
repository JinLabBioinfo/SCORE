#!/bin/bash

source $HOME/.bashrc
source $HOME/.bash_profile

MATDIR=$1
RESOL=$2
BIN=$3
OUTDIR=$4
LIBID=$5

mkdir -p $OUTDIR

LIBIDs=(${LIBID})

for LIBNAME in "${LIBIDs[@]}"; do

# see this: https://unix.stackexchange.com/questions/426748/cat-a-very-large-number-of-files-together-in-correct-order
find $MATDIR/ -maxdepth 1 -type f -name \*.sparse.matrix_${BIN} -print0 |
sort -zV |
xargs -0 cat > $OUTDIR/sparse_matrices_${LIBNAME}_${BIN}.txt

done
