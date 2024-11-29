import os

import s3fs
from osgeo import gdal
from pqdm.processes import pqdm


def convert_jp2_to_geotiff(file: str):
    """
    Converts a JP2 image to GeoTIFF format and saves it to a specified S3 path.

    Args:
        file (str): The path to the JP2 image file.

    Returns:
        dict: A dictionary containing the result of the conversion and the path of the output file.
            The dictionary has the following keys:
            - "result" (str): The result of the conversion, either "OK" or "FAILED".
            - "file" (str): The path of the output file.

    Raises:
        Exception: If there is an error opening the JP2 image file.

    """
    try:
        # Open the .jp2 image
        ds = gdal.Open(f"/vsis3/{file}")
    except Exception as e:
        print(f"Failed to open {file} with error : {e}")
        return {"result": "FAILED", "file": file}

    # Get the driver for GeoTIFF format
    driver = gdal.GetDriverByName("GTiff")

    # Specify the S3 path where you want to save the file
    file_output = file.replace(".jp2", ".tif")
    driver.CreateCopy(f"/vsis3/{file_output}", ds)
    # Close the datasets
    ds = None
    return {"result": "OK", "file": "/".join(file_output.split("/")[-3:])}


gdal.UseExceptions()
os.environ["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"})

list_jp2 = [
    "/".join(file.split("/")[-3:])
    for file in fs.glob("projet-slums-detection/data-raw/PLEIADES/**/**/*.jp2")
]

list_tif = [
    "/".join(file.split("/")[-3:])
    for file in fs.glob("projet-slums-detection/data-raw/PLEIADES/**/**/*.tif")
]

list_already_converted = [
    f"projet-slums-detection/data-raw/PLEIADES/{file}"
    for file in list_jp2
    if file.replace(".jp2", ".tif") in list_tif
]

list_converted = []

while len(list_jp2) > len(list_converted + list_already_converted):
    file_missing = [
        f"projet-slums-detection/data-raw/PLEIADES/{file}"
        for file in list_jp2
        if file.replace(".jp2", ".tif") not in list_converted + list_already_converted
    ]
    result = pqdm(file_missing, convert_jp2_to_geotiff, n_jobs=50)
    for i in range(len(result)):
        if isinstance(result[i], RuntimeError):
            continue
        else:
            if result[i]["result"] != "FAILED":
                list_converted.append(result[i]["file"])
