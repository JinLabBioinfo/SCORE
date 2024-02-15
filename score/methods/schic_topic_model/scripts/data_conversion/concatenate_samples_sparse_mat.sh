#!/bin/bash

source $HOME/.bashrc
source $HOME/.bash_profile

MATDIR=$1
BIN=$2
LIBID=$3
BINNAME=`expr $BIN \* 2`

LIBIDs=(${LIBID})
COUNT=0

## create output file
OUTFILE=$MATDIR/all_sparse_matrices_${BIN}.txt
touch $OUTFILE

## add to output file
for LIBNAME in "${LIBIDs[@]}"; do

awk -v count="$COUNT" '{print ($1+count), '\t', $2, '\t', $3}' $MATDIR/sparse_matrices_${LIBNAME}_${BIN}.txt | cat >> $OUTFILE

NUMCELLS=`awk 'END{print $1}' $MATDIR/sparse_matrices_${LIBNAME}_${BIN}.txt`
COUNT=`expr $COUNT + $NUMCELLS`

done
