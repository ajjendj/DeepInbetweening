#!/bin/sh -e

set -o nounset

input=$1
output_dir=$2

DIR_PATH=/Users/ajjoshi/Documents/Data/classic/

ls $input
echo "Input is $input"

for x in $(cat $input); do
echo $x
python -m preprocess.triplets_data_main \
  --output_dir=$output_dir \
  --video=$DIR_PATH$x #\
#  --synthetic_motion='AFFINE'
done