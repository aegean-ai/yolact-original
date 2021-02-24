#!/bin/bash

start=`date +%s`

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ./data
    echo "navigating to ./data/ ..."
    mkdir -p ./data
    cd ./data/
    mkdir -p ./coco
    cd ./coco
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
cd ./images
#echo "Downloading MSCOCO train images ..."
#curl -LO http://images.cocodataset.org/zips/train2017.zip
echo "Downloading MSCOCO train images from S3"
aws s3 cp s3://cv.datasets.aegean.ai/mscoco/train2017.zip .

#echo "Downloading MSCOCO val images ..."
#curl -LO http://images.cocodataset.org/zips/val2017.zip
echo "Downloading MSCOCO val images from S3"
aws s3 cp s3://cv.datasets.aegean.ai/mscoco/val2017.zip .


cd ../
if [ ! -d annotations ]
  then
    mkdir -p ./annotations
fi

# Download the annotation data.
cd ./annotations
#echo "Downloading MSCOCO train/val annotations ..."
#curl -LO http://images.cocodataset.org/annotations/annotations_trainval2014.zip
#curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
#echo "Finished downloading. Now extracting ..."
echo "Downloading MSCOCO train/val annotations from s3" 
# Skipping 2014 as there is no corresponding file in s3 
aws s3 cp s3://cv.datasets.aegean.ai/mscoco/annotations_trainval2017.zip .
echo "Finished downloading. Now extracting"

# Unzip data
echo "Extracting train images ..."
unzip -qqjd ../images ../images/train2017.zip
echo "Extracting val images ..."
unzip -qqjd ../images ../images/val2017.zip
#echo "Extracting annotations ..."
#Skipped as file isn't available
#unzip -qqd .. ./annotations_trainval2014.zip
unzip -qqd .. ./annotations_trainval2017.zip

echo "Removing zip files ..."
rm ../images/train2017.zip
rm ../images/val2017.zip
#Skipped as file isn't available
#rm ./annotations_trainval2014.zip
rm ./annotations_trainval2017.zip


end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"
