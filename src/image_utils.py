import re
from typing import List
from pyproj import Transformer


def find_image_of_point(
    coordinates: List,
    list_filepaths: str,
    crs: int,
    coord_gps: bool = True,
) -> str:
    """
    Gives the image in the folder which contains the point (gps or crs).
    This method is based on the filenames of the pleiades images.
    Returns a message if the image is not in the folder.

    Args:
        coordinates (List):
            [x,y] CRS coordinate or [lat, lon] gps coordinate
        folder_path (str):
            The path of the folder in which we search the image containing
            the point.
        coord_gps (boolean):
            Specifies if the coordinate is a gps coordinate or not.

    Returns:
        str:
            The path of the image containing the point.

    Examples:
        >>> point = (-12.768023, 45.190708)
        >>> image_path = find_image_of_point(point, list_images_mayotte_2020, 4471, True)
    """

    if coord_gps:
        # Retrieve the crs via the department
        lat, lon = coordinates
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{str(crs)}", always_xy=True)
        x, y = transformer.transform(lon, lat)

    else:
        x, y = coordinates

    # Retrieve left-top coordinates
    delimiters = ["-", "_"]

    pattern = "|".join(delimiters)

    for filename in list_filepaths:
        filename = filename.split("/")[-1]
        split_filename = re.split(pattern, filename)

        left = float(split_filename[2]) * 1000
        top = float(split_filename[3]) * 1000
        right = left + 1000.0
        bottom = top - 1000.0

        if left <= x <= right:
            if bottom <= y <= top:
                return filename
    else:
        return "The point is not find in the folder."
