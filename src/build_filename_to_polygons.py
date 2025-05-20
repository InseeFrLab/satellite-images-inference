import geopandas as gpd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
from astrovision.data import SatelliteImage
from osgeo import gdal
from pqdm.processes import pqdm
from shapely import Polygon

gdal.UseExceptions()


def create_polygon(image: str) -> gpd.GeoDataFrame:
    try:
        si = get_satellite_image(image, 3)
    except Exception as e:
        print(f"Error loading satellite image: {e}")
        # Create an empty GeoDataFrame with no geometry
        gdf = gpd.GeoDataFrame()
        gdf["filename"] = image
        gdf["CRS"] = None
        gdf["year"] = image.split("/")[-2]
        gdf["dep"] = image.split("/")[-3]
        return gdf

    # Create a polygon from the bounds
    minx, miny, maxx, maxy = si.bounds
    polygon = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

    # Create a GeoDataFrame with the polygon
    gdf = gpd.GeoDataFrame(geometry=[polygon], crs=si.crs)
    gdf["filename"] = image
    gdf["CRS"] = si.crs
    gdf["year"] = image.split("/")[-2]
    gdf["dep"] = image.split("/")[-3]
    return gdf


def get_satellite_image(image_path: str, n_bands: int):
    """
    Retrieves a satellite image specified by its path.

    Args:
        image_path (str): Path to the satellite image.
        n_bands (int): Number of bands in the satellite image.

    Returns:
        SatelliteImage: An object representing the satellite image.
    """

    # Load satellite image using the specified path and number of bands
    si = SatelliteImage.from_raster(
        file_path=f"/vsis3/{image_path}",
        dep=None,
        date=None,
        n_bands=n_bands,
    )
    return si


fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"})

list_filename = [f for d in ["MAYOTTE", "MARTINIQUE", "GUADELOUPE", "REUNION", "GUYANE"] for f in fs.glob(f"projet-slums-detection/data-raw/PLEIADES/{d}/**/*.tif")]

current_filename_to_poly = (
    pq.ParquetDataset("projet-slums-detection/data-raw/PLEIADES/filename-to-polygons/", filesystem=fs).read().to_pandas()
)

list_missing_files = [x for x in list_filename if x not in current_filename_to_poly.filename.to_list()]

file_retrieved = []
list_gpd = []

while len(list_missing_files) > len(file_retrieved):
    file_missing = [file for file in list_missing_files if file not in file_retrieved]
    result = pqdm(file_missing, create_polygon, n_jobs=50)
    for i in range(len(result)):
        tmp = result[i]
        if not result[i]["CRS"].empty:
            tmp.crs = None
            file_retrieved.append(tmp.loc[0, "filename"])
            list_gpd.append(tmp)


merged_gdf = gpd.GeoDataFrame(pd.concat(list_gpd, ignore_index=True))
merged_gdf["geometry"] = merged_gdf.geometry.to_wkt()
table = pa.Table.from_pandas(merged_gdf)

pq.write_to_dataset(
    table,
    root_path="s3://projet-slums-detection/data-raw/PLEIADES/filename-to-polygons",
    partition_cols=["dep", "year"],
    basename_template="part-{i}.parquet",
    existing_data_behavior="overwrite_or_ignore",
    filesystem=fs,
)
