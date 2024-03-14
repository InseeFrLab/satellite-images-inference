import requests
import numpy as np
from astrovision.data import SatelliteImage, SegmentationLabeledSatelliteImage
from astrovision.plot import plot_images_with_segmentation_label
import os
from s3fs import S3FileSystem
from osgeo import gdal, ogr, osr
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import rasterize, shapes
from shapely.geometry import Polygon, shape
import rasterio



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


# Polygoniser les amas de 1 -> un amas de 1 = batiment
array = lsi.label
left, __, __, top = lsi.satellite_image.bounds

array = array.astype(np.uint16)

with rasterio.Env():
    with rasterio.open('temp.tif', 'w+', driver='GTiff', 
                       height=array.shape[0], width=array.shape[1],
                       count=1, dtype=array.dtype, nodata=0,
                       transform=rasterio.transform.from_origin(left, top, 0.5, 0.5)) as dst:
        dst.write(array, 1)

        results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) in enumerate(shapes(array, mask=None, transform=dst.transform))
            if v == 1  # Conserver uniquement les amas de 1
        )

gdf = gpd.GeoDataFrame.from_features(list(results))
gdf.crs = lsi.satellite_image.crs
gdf.to_parquet(f"results/polygons/{filename}_polygons-geo.parquet")

# supprimer le fichier temp.tif
if os.path.exists('temp.tif'):
    os.remove('temp.tif')

# Afficher les polygones

gdf = gpd.read_parquet(f"results/polygons/{filename}_polygons-geo.parquet")
print(gdf.head())

fig, ax = plt.subplots()
ax = gdf.plot(facecolor='blue', edgecolor='black')
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


# aire des batiments
union_bat = gdf['geometry'].unary_union

total_area_bat = union_bat.area

# comme un pixel = 0.5*0.5m²
total_area = lsi.satellite_image.array.shape[1]*0.5*lsi.satellite_image.array.shape[2]*0.5

## ou on peut aussi faire le polygone image
# polygon = Polygon([(left, bottom), (left, top), (right, top), (right, bottom), (left, bottom)])
# tot = gpd.GeoDataFrame([1], geometry=[polygon], crs=lsi.satellite_image.crs)
# total_area = tot.geometry.area
# print(total_area)

prop_bat = total_area_bat*100/total_area
# 25.5 % de l'image est du bâtiment
