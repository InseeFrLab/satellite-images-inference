import s3fs
from pqdm.processes import pqdm
from osgeo import gdal
import os


def convert_jp2_to_geotiff(file):
    try:
        # Open the .jp2 image
        ds = gdal.Open(f"/vsis3/{file}")
    except Exception as e:
        print(f"Failed to open {file} with error : {e}")
        return {"result": "FAILED", "file": file}

    # Get the driver for GeoTIFF format
    driver = gdal.GetDriverByName("GTiff")

    # Specify the S3 path where you want to save the file
    file_output = file.replace("data-raw", "data-raw-tif").replace(".jp2", ".tif")
    driver.CreateCopy(f"/vsis3/{file_output}", ds)
    # Close the datasets
    ds = None
    return {"result": "OK", "file": "/".join(file_output.split("/")[-3:])}


gdal.UseExceptions()
os.environ["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"})

list_filename = [
    "/".join(file.split("/")[-3:])
    for file in fs.glob("projet-slums-detection/data-raw/PLEIADES/**/**/*.jp2")
]

file_retrieved = [
    "/".join(file.split("/")[-3:])
    for file in fs.glob("projet-slums-detection/data-raw-tif/PLEIADES/**/**/*.tif")
]

while len(list_filename) > len(file_retrieved):
    file_missing = [
        f"projet-slums-detection/data-raw/PLEIADES/{file}"
        for file in list_filename
        if file.replace(".jp2", ".tif") not in file_retrieved
    ]
    result = pqdm(file_missing, convert_jp2_to_geotiff, n_jobs=50)
    for i in range(len(result)):
        if isinstance(result[i], RuntimeError):
            continue
        else:
            if result[i]["result"] != "FAILED":
                file_retrieved.append(result[i]["file"])
