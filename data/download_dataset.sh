#!/bin/bash

wget  --no-clobber http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz 
tar xvzf gdb9.tar.gz
rm gdb9.tar.gz
rm gdb9.sdf.csv

if [ "gdb9.sdf" != "$1" ]; then
   mv gdb9.sdf $1
fi

# Download the score files 
# Hide progress bar so it doesn't pollute the CI logs
wget -nv --no-clobber https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
wget -nv --no-clobber https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz

# Rename the downloaded files
if [ "NP_score.pkl.gz" != "$2" ]; then
   mv NP_score.pkl.gz $2
fi

if [ "SA_score.pkl.gz" != "$3" ]; then
   mv SA_score.pkl.gz $3
fi
