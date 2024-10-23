#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <departement> <annee> <model> <version> <nom_pod>"
    echo "Example: $0 MAYOTTE 2023 Segmentation 1 geoserver-pod-0"
    echo "Note: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_S3_ENDPOINT must be set as environment variables"
    exit 1
fi

# Check if required environment variables are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$AWS_S3_ENDPOINT" ]; then
    echo "Error: AWS environment variables not set"
    echo "Please set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT"
    exit 1
fi

# Assign arguments to variables
departement="$1"
annee="$2"
model="$3"
version="$4"
nom_pod="$5"


# Install MinIO client
echo "Installing MinIO client if needed..."
kubectl exec ${nom_pod} -c geoserver -- /bin/bash -c "\
    if ! command -v mc &> /dev/null; then \
        wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O /usr/local/bin/mc && \
        chmod +x /usr/local/bin/mc && \
        echo 'MinIO client installed.' \
    else \
        echo 'MinIO client is already installed.' \
    fi"

# Copy raw data from S3 to the GeoServer directly
echo "Copying raw data..."
kubectl exec ${nom_pod} -c geoserver -- /bin/bash -c "\
    mc cp -r s3/projet-slums-detection/data-raw/PLEIADES/${departement}/${annee}/  \
        /opt/geoserver/data_dir/PLEIADES/${departement}/${annee} && \
    echo 'Raw data successfully copied.'"

# Copy predictions from S3 to the GeoServer directly
echo "Copying predictions..."
kubectl exec ${nom_pod} -c geoserver -- /bin/bash -c "\
    export MC_HOST_s3=https://${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}@${AWS_S3_ENDPOINT} && \
    mc cp s3/projet-slums-detection/data-prediction/PLEIADES/${departement}/${annee}/${model}/${version}/predictions.gpkg \
        /opt/geoserver/data_dir/PREDICTIONS/PLEIADES/${departement}/${annee}/${model}/${version}/predictions.gpkg && \
    echo 'Predictions file successfully copied.'"

echo "Script completed successfully!"