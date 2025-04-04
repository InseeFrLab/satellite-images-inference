#!/bin/bash

sudo apt-get update
sudo apt-get install lftp
sudo apt-get install p7zip-full

# FTP server details
FTP_SERVER="ftp.seasguyane.org"
FTP_USER="insee973"
FTP_PASS="********"

IMAGE="DS_PHR1A_202306081414359_GU1_PX_W055N05_1008_01654-F-D.zip"
lftp -e "set ftp:ssl-force true; set ssl:verify-certificate no; open ftp://$FTP_SERVER; user $FTP_USER $FTP_PASS;
cd Arch-2024-042/guyane/2023/F/D

mget $IMAGE

bye"

sha1sum $IMAGE

# repair zip (usually not good practice, means that the download leads to corrupted file, you would prefer retry the download)
zip -FF DS_PHR1A_202301241402156_FR1_PX_W054N04_0622_00596-F-D.zip --out DS_PHR1A_202301241402156_FR1_PX_W054N04_0622_00596-F-D.zip

# list zip
7z l DS_PHR1A_202306081414359_GU1_PX_W055N05_1008_01654-F-D.zip

# unzip
7z x DS_PHR1A_202306081414359_GU1_PX_W055N05_1008_01654-F-D.zip


# Create the destination directory if it doesn't exist
dest_dir="PLEIADES/GUYANE/2023"

mkdir -p "$dest_dir"

# Move all tif file to a specific folder before sending data to minio
find . -type f -name "*.TIF" | while read -r file; do
    # Rename the file to .tif
    new_file="${file%.TIF}.tif"
    mv "$file" "$new_file"
    # Move the renamed file to the destination directory
    mv "$new_file" "$dest_dir/"
done

mc cp -r $dest_dir/ s3/projet-slums-detection/data-raw/$dest_dir/
