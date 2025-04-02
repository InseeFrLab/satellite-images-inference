# Si beaucoup d'images, penser à ouvrir un service avec une persistance élevée (ex size: 500Gi) via l'éditeur de texte, et le fermer une fois l'opération finie.

#!/bin/bash

sudo apt-get update
sudo apt-get install lftp
sudo apt-get install p7zip-full

# FTP server details
FTP_SERVER="ftp3.ign.fr"
FTP_USER="INSEE"
FTP_PASS="*********"

lftp -e "open ftp://$FTP_SERVER; user $FTP_USER $FTP_PASS;" 

# Récuperer le dossier en local
ls MAYOTTE/
mirror 2025

# Convertir les images en TIFF pour ensuite les stocker sur le s3 avec le code write_jp2_to_tiff.py