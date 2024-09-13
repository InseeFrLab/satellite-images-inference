departement="SAINT-MARTIN"
annee="2024"
nom_pod="geoserver-pod-0"
model="test"
version="15"

# Install MinIO client (Ne faire que si le pod vient de redémarrer, une fois installé ça sert à rien de le réinstaller à chaque fois)
kubectl exec ${nom_pod} -c geoserver -- /bin/bash -c "\
    wget -q https://dl.min.io/client/mc/release/linux-amd64/mc -O /usr/local/bin/mc && \
    chmod +x /usr/local/bin/mc && \
    echo 'MinIO client installed.'"

# Copy raw data from S3 to the GeoServer directly
kubectl exec ${nom_pod} -c geoserver -- /bin/bash -c "\
    mc cp -r s3/projet-slums-detection/data-raw/PLEIADES/${departement}/${annee}/  \
        /opt/geoserver/data_dir/PLEIADES/${departement}/${annee} && \
    echo 'Raw data successfully copied.'"

# Copy predictions from S3 to the GeoServer directly
kubectl exec ${nom_pod} -c geoserver -- /bin/bash -c "\
    export MC_HOST_s3=https://${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY}@${AWS_S3_ENDPOINT} && \
    mc cp s3/projet-slums-detection/data-prediction/PLEIADES/${departement}/${annee}/${model}/${version}/predictions.gpkg \
        /opt/geoserver/data_dir/PREDICTIONS/PLEIADES/${departement}/${annee}/${model}/${version}/predictions.gpkg && \
    echo 'Predictions file successfully copied.'"

# Ensuite création de la couche image mosaique à partir des images tifs injectées dans le geoserver (àla main)
# Création de la couche predictions.gpkg
# Mise à disposition sur Cratt

# pourfaire un peu de nettoyage dans lesystme de fichier du geoserver
kubectl exec -it ${nom_pod} -c geoserver -- /bin/bash
