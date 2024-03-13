import requests
import numpy as np
from astrovision.data import SatelliteImage, SegmentationLabeledSatelliteImage
import os
from astrovision.plot import plot_images_with_segmentation_label
from s3fs import S3FileSystem
from osgeo import gdal, ogr, osr
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import rasterize

os.makedirs(
        "results/",
        exist_ok=True,
    )
os.makedirs(
    "results/polygons/",
    exist_ok=True,
)
os.makedirs(
    "results/plots/",
    exist_ok=True,
)

image_path = "projet-slums-detection/golden-test/patchs/segmentation/PLEIADES/MAYOTTE_CLEAN/2022/250/ORT_976_2022_0524_8587_U38S_8Bits_0011.jp2"
# image_path = "projet-slums-detection/data-preprocessed/patchs/BDTOPO/segmentation/PLEIADES/GUADELOUPE/2022/250/train/ORT_971_2022_0627_1797_U20N_8bits_0037.jp2"
# image_path = "projet-slums-detection/data-preprocessed/patchs/BDTOPO/segmentation/PLEIADES/GUADELOUPE/2022/250/train/ORT_971_2022_0627_1799_U20N_8bits_0049.jp2"
# image_path = "projet-slums-detection/data-preprocessed/patchs/BDTOPO/segmentation/PLEIADES/GUADELOUPE/2022/250/train/ORT_971_2022_0627_1801_U20N_8bits_0042.jp2"

response = requests.get(
    "https://satellite-images-inference.lab.sspcloud.fr/predict", params={"image": image_path}
)
filename = image_path.split("/")[-1].split(".")[0]

lsi = SegmentationLabeledSatelliteImage(
    SatelliteImage.from_raster(
        file_path=f"/vsis3/{image_path}",
        dep=None,
        date=None,
        n_bands=3,
    ),
    np.array(response.json()["mask"]),
)
plot = lsi.plot(bands_indices=[0, 1, 2])
plot.savefig(f"results/plots/{filename}_api_result.png")

# Polygoniser les amas de 1
array = lsi.label
top_left_x, bottom_right_y, bottom_right_x, top_left_y = lsi.satellite_image.bounds

driver = gdal.GetDriverByName('GTiff')
outRaster = driver.Create('temp.tif', 250, 250, 1, gdal.GDT_Byte)
outRaster.GetRasterBand(1).WriteArray(array)

# (coin_x, taille_pixel_x, rotation_x, coin_y, rotation_y, taille_pixel_y)
outRaster.SetGeoTransform((top_left_x, 0.5, 0, top_left_y, 0, -0.5))

# Créer le fichier pour les polygones
outShapefile = f"results/polygons/{filename}_polygons.shp"
outDriver = ogr.GetDriverByName("ESRI Shapefile")
outDataSource = outDriver.CreateDataSource(outShapefile)
outLayer = outDataSource.CreateLayer("polygons", geom_type=ogr.wkbPolygon)
idField = ogr.FieldDefn("id", ogr.OFTInteger)
outLayer.CreateField(idField)

# Polygoniser
gdal.Polygonize(outRaster.GetRasterBand(1), None, outLayer, 0, [], callback=None)

# Fermer le fichier raster et shapefile
outRaster = None
outDataSource = None

# Afficher les polygones
gdf = gpd.read_file(f"results/polygons/{filename}_polygons.shp")
fig, ax = plt.subplots()
ax = gdf.plot()
plt.xticks([])
plt.yticks([])
plt.savefig(f'results/plots/{filename}_polygons.png')


# Afficher les polygones sur l'image satellite
left, bottom, right, top = lsi.satellite_image.bounds

fig, ax = plt.subplots()
plt.imshow(np.transpose(lsi.satellite_image.array, (1, 2, 0))[:, :, [0, 1, 2]]
, extent=[left, right, bottom, top])

gdf.plot(ax=ax, facecolor='none', edgecolor='red')

plt.xticks([])
plt.yticks([])

plot = plt.gcf()
plot.savefig(f'results/plots/{filename}_image_and_polygon.png')
plt.close()
