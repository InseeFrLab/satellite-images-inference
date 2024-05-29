departement="GUYANE"
annee="2023"
nom_pod="ggg-geoserver-68f45b8854-v6cbz"
model="test"
version="16"

mc cp --recursive s3/projet-slums-detection/data-raw-tif/PLEIADES/${departement}/${annee}/ ${departement}/${annee}
kubectl exec ${nom_pod} -- mkdir -p /opt/geoserver/data_dir/PLEIADES/${departement}/${annee}
kubectl cp  ${departement}/${annee}/ projet-slums-detection/${nom_pod}:/opt/geoserver/data_dir/PLEIADES/${departement}/${annee}/
kubectl exec ${nom_pod} -- chmod -R a+rw ../opt/geoserver/data_dir

mc cp s3/projet-slums-detection/data-prediction/PLEIADES/${departement}/${annee}/${model}/${version}/predictions.gpkg predictions.gpkg
kubectl exec ${nom_pod} -- mkdir -p /opt/geoserver/data_dir/PREDICTIONS/PLEIADES/${departement}/${annee}/${model}/${version}
kubectl cp  predictions.gpkg projet-slums-detection/${nom_pod}:/opt/geoserver/data_dir/PREDICTIONS/PLEIADES/${departement}/${annee}/${model}/${version}/predictions.gpkg
kubectl exec ${nom_pod} -- chmod -R a+rw ../opt/geoserver/data_dir


# Ensuite création de la couche image mosaique à partir des images tifs injectées dans le geoserver (àla main)
# Création de la couche predictions.gpkg
# Mise à disposition sur Cratt  


# pourfaire un peu de nettoyage dans lesystme de fichier du geoserver
kubectl exec -it ${nom_pod} -- /bin/bash 