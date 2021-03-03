#!/bin/bash

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ./data
    echo "navigating to ./data/ ..."
    mkdir -p ./data
    cd ./data/
    mkdir -p ./dvrpc
    cd ./dvrpc
    mkdir -p ./images
    mkdir -p ./annotations
  else
    # check if specified dir is valid
    if [ ! -d $1 ]; then
        echo $1 " is not a valid directory"
        exit 0
    fi
    echo "navigating to " $1 " ..."
    cd $1
fi

if [ ! -d images ]
  then
    mkdir -p ./images
fi

# Download the image data.
echo "Downloading DVRPC train and test images from S3"
aws s3 cp s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/dvrpc-pedestrian-network-pa-only-2020/train.tar.gz .
aws s3 cp s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/dvrpc-pedestrian-network-pa-only-2020/test.tar.gz .

if [ ! -d annotations ]
  then
    mkdir -p ./annotations
fi

# Download the annotation data.
cd ./annotations
echo "Downloading DVRPC train and test annotations from s3" 
aws s3 cp s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/dvrpc-pedestrian-network-pa-only-2020/DVRPC_train.json .
aws s3 cp s3://cv.datasets.aegean.ai/njtpa/njtpa-year-2/dvrpc-pedestrian-network-pa-only-2020/DVRPC_test.json .
echo "Finished downloading."


cd ../images
echo "Extracting test images into the image folder"
tar xf ../test.tar.gz  --strip-components=1
echo "Extracting train images into the image folder"
tar xf ../train.tar.gz  --strip-components=1
echo "Finished Extraction"

#echo "Removing zip files"
#rm ../test.tar.gz 
#rm ../train.tar.gz

end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"
