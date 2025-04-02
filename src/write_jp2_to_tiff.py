import os

import s3fs
from osgeo import gdal
from pqdm.processes import pqdm


def convert_jp2_to_geotiff(args):
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
    file_input, file_output = args

    try:
        # Open the .jp2 image
        ds = gdal.Open(f"{file_input}")
    except Exception as e:
        print(f"Failed to open {file_input} with error : {e}")
        return {"result": "FAILED", "file": file_input}

    # Get the driver for GeoTIFF format
    driver = gdal.GetDriverByName("GTiff")

    # Specify the S3 path where you want to save the file
    driver.CreateCopy(f"/vsis3/{file_output}", ds)
    # Close the datasets
    ds = None
    return {"result": "OK", "file": file_output}


gdal.UseExceptions()
os.environ["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"

fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"})

parent = os.path.abspath("..")
dossier_images_jp2_local = parent+"/2025/MOSA"  # a modifier
destination_s3 = "MAYOTTE/2025"  # a modifier

list_jp2 = [
    file
    for file in os.listdir(dossier_images_jp2_local)
    if file.endswith(".jp2")
]

list_converted = []

while len(list_jp2) > len(list_converted):
    file_missing_input = [
        f"{dossier_images_jp2_local}/{file}"
        for file in list_jp2
        if file.replace(".jp2", ".tif") not in list_converted
    ]
    file_missing_output = [
        f"projet-slums-detection/data-raw/PLEIADES/{destination_s3}/{file.replace(".jp2", ".tif")}"
        for file in list_jp2
        if file.replace(".jp2", ".tif") not in list_converted
    ]
    result = pqdm(zip(file_missing_input, file_missing_output), convert_jp2_to_geotiff, n_jobs=50)
    for i in range(len(result)):
        if isinstance(result[i], RuntimeError):
            continue
        else:
            if result[i]["result"] != "FAILED":
                list_converted.append(result[i]["file"].split('/')[-1])
