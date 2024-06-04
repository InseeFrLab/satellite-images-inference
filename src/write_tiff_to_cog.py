import s3fs
from pqdm.processes import pqdm
from osgeo import gdal
import os

from rasterio.io import MemoryFile
import boto3
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import subprocess

gdal.UseExceptions()


def convert_geotiff_to_cog(file: str):
    """
    Converts a GeoTIFF file to a Cloud-Optimized GeoTIFF (COG) format.

    Args:
        file (str): The path of the GeoTIFF file to be converted.

    Returns:
        dict: A dictionary containing the result of the conversion. The dictionary has two keys:
            - 'result': The result of the conversion, which can be either 'OK' or 'FAILED'.
            - 'file': The path of the converted COG file.

    Raises:
        Exception: If an error occurs during the conversion process.
    """
    try:
        # Check that the file is a COG first
        result = int(
            subprocess.run(
                [
                    "python",
                    "src/validate_cog.py",
                    "--filename",
                    f"/vsis3/projet-slums-detection/{file}",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        if result == 0:
            # File is already a COG
            return {"result": "OK", "file": file}
        else:
            with MemoryFile() as mem_dst:
                # Important, we pass `mem_dst.name` as output dataset path
                cog_translate(
                    f"/vsis3/projet-slums-detection/{file}",
                    mem_dst.name,
                    cog_profiles.get("lzw"),
                    in_memory=True,
                    quiet=True,
                )
                client = boto3.client(
                    "s3",
                    endpoint_url="https://" + "minio.lab.sspcloud.fr",
                    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                )
                client.upload_fileobj(mem_dst, "projet-slums-detection", f"/{file}")

                # Check that the file have been translated well
                result = int(
                    subprocess.run(
                        [
                            "python",
                            "src/validate_cog.py",
                            "--filename",
                            f"/vsis3/projet-slums-detection/{file}",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                )
                if result == 0:
                    # It is OK
                    return {"result": "OK", "file": file}
                else:
                    return {"result": "FAILED", "file": file}

    except Exception as e:
        print(f"Failed to translate {file} with error : {e}")
        return {"result": "FAILED", "file": file}


fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"})

list_tif = [
    "/".join(file.split("/")[-5:])
    for file in fs.glob("projet-slums-detection/data-raw/PLEIADES/**/**/*.tif")
]

list_converted = []

while len(list_tif) > len(list_converted):
    file_missing = [file for file in list_tif if file not in list_converted]
    result = pqdm(file_missing, convert_geotiff_to_cog, n_jobs=50)
    for i in range(len(result)):
        if isinstance(result[i], RuntimeError):
            continue
        else:
            if result[i]["result"] == "OK":
                list_converted.append(result[i]["file"])
