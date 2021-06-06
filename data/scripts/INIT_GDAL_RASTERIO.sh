#!/bin/bash

# the docker container under docker/yolcat-original installs the GDAL - the script below is redudant for docker development but it is kept FYI. 

sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update
sudo apt-get install gdal-bin libgdal-dev
pip install -U pip
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')
pip install rasterio