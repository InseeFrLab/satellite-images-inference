kubectl exec -it geoserver-pod-0 -- /bin/bash

wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O /tmp/mc 
chmod +x /tmp/mc 

export MC_HOST_s3=

/tmp/mc cp s3/projet-slums-detection/data-raw/PLEIADES/GUYANE/2024/ /opt/geoserver/data_dir/PLEIADES/GUYANE/2024
chmod -R a+rw /opt/geoserver/data_dir
