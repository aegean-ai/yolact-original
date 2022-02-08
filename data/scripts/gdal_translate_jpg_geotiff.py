import os

os.system("""
for f in /workspaces/data/njtpa.auraison.aegean.ai/dvrpc-pedestrian-network-pa-only-2020/Train/*.jpg;
    do gdal_translate -of GTiff -co COMPRESS=JPEG -co JPEG_QUALITY=90 -co PHOTOMETRIC=YCBCR $f ${f%.*}.tif
done
""")