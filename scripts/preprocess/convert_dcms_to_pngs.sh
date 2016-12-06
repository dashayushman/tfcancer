#! /bin/bash

# Generates all .png files
#
# To be called from the folder where the DOI dataset was downloaded
#
# `convert` is an `imagemagick` tool

for f in ISPY*/*/*/*.dcm ; do
	#files=`ls $d | wc -l`

	#for ((i = 0; i < $files; i++));
	#for f in `ls $d`
	#do
    png="${f/dcm/png}"
    echo "convert $f $png"
    convert $f $png
	#done
done

