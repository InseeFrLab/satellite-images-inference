import re
from typing import List
from pyproj import Transformer
from shapely.geometry import box, Point


def find_image_of_point(
    coordinates: List(float),
    list_filepaths: List(str),
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
        >>> point = (-12.783007, 45.221377)
        >>> image_path = find_image_of_point(point, list_images_mayotte_2020, 4471, True)
        'ORT_2020052526670967_0524_8587_U38S_8Bits'
    """

    if coord_gps:
        # Retrieve the crs via the department
        lat, lon = coordinates
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{str(crs)}", always_xy=True)
        x, y = transformer.transform(lon, lat)

    else:
        x, y = coordinates

    point = Point(x, y)

    delimiters = ["-", "_"]
    pattern = "|".join(delimiters)

    for filename in list_filepaths:
        filename = filename.split("/")[-1]
        split_filename = re.split(pattern, filename)

        left = float(split_filename[2]) * 1000
        top = float(split_filename[3]) * 1000
        right = left + 1000.0
        bottom = top - 1000.0
        bbox = box(left, bottom, right, top)

        # Vérifier si le point est inclus dans la bounding box
        if bbox.contains(point):
            return filename

    return "The point is not find in the folder."


def find_images_of_bb(
    bbox: box,
    list_filepaths: List(str),
) -> List(str):
    """
    Gives the images in the folder which are in the bounding box.
    This method is based on the filenames of the pleiades images.

    Args:
        bbox (box):
            bounding box in the CRS of the isle.
        folder_path (str):
            The path of the folder in which we search the images contained in
            the bounding box.

    Returns:
        List(str):
            The list of the images paths included in the bounding box.

    Examples:
        >>> left, bottom, right, top = (523500.0, 8586500.0, 524500.0, 8587500.0)
        >>> bounding_box = box(left, bottom, right, top)
        >>> find_images_of_bb(bounding_box, list_images_mayotte_2020)
        ['ORT_2020052526670967_0523_8587_U38S_8Bits.jp2',
        'ORT_2020052526670967_0523_8588_U38S_8Bits.jp2',
        'ORT_2020052526670967_0524_8587_U38S_8Bits.jp2',
        'ORT_2020052526670967_0524_8588_U38S_8Bits.jp2']
    """
    # Retrieve left-top coordinates
    delimiters = ["-", "_"]

    pattern = "|".join(delimiters)

    filepaths_images_in_bb = []
    for filename in list_filepaths:
        filename = filename.split("/")[-1]
        split_filename = re.split(pattern, filename)

        left = float(split_filename[2]) * 1000
        top = float(split_filename[3]) * 1000
        right = left + 1000.0
        bottom = top - 1000.0

        bbox2 = box(left, bottom, right, top)

        if bbox.intersects(bbox2):
            filepaths_images_in_bb.append(filename)

    return filepaths_images_in_bb
