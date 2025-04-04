from typing import Union

import geopandas as gpd
import libpysal
import networkx as nx
import pandas as pd
from geopandas import GeoSeries
from shapely.geometry import LineString, MultiLineString, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm


def check_line_intersection(poly1: Union[LineString, MultiLineString], poly2: Union[LineString, MultiLineString]) -> bool:
    """
    Check if two lines intersect.

    Args:
        poly1 (Polygon): The first line.
        poly2 (Polygon): The second line.

    Returns:
        bool: True if the lines intersect, False otherwise.
    """
    intersection = poly1.intersection(poly2)
    return isinstance(intersection, (LineString, MultiLineString))


def merge_adjacent_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Merge adjacent polygons in a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to process.

    Returns:
        gpd.GeoDataFrame: The GeoDataFrame with adjacent polygons merged.
    """
    # TODO: potentially add tiny buffer on polygons ?
    # Create a spatial weights matrix
    W = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    # Merge adjacent polygons
    components = W.component_labels
    merged_gdf = gdf.dissolve(by=components)
    return merged_gdf


def clean_prediction(gdf_original: gpd.GeoDataFrame, buffer_distance: int = 3) -> gpd.GeoDataFrame:
    """
    Clean the predictions by merging adjacent polygons and removing small artifacts.

    Args:
        gdf_original (gpd.GeoDataFrame): The original GeoDataFrame containing the predictions.
        buffer_distance (float, optional): The buffer distance to apply to the polygons. Defaults to 1.5.

    Returns:
        gpd.GeoDataFrame: The cleaned GeoDataFrame.

    """
    gdf = merge_adjacent_polygons(gdf_original.copy())

    # Buffer the geometry
    gdf["geometry"] = gdf["geometry"].buffer(-buffer_distance).buffer(buffer_distance)
    sindex = gdf.sindex

    # Find touching pairs
    touching_pairs = set()
    for idx, poly in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Processing geometries"):
        neighbors_indices = set(sindex.query(poly.geometry, predicate="touches"))
        touching_pairs.update(
            (min(idx, neighbor_idx), max(idx, neighbor_idx))
            for neighbor_idx in neighbors_indices
            if neighbor_idx != idx and check_line_intersection(poly.geometry, gdf.iloc[neighbor_idx].geometry)
        )

    # Create a connected graph
    G = nx.Graph(touching_pairs)
    connected_components = list(nx.connected_components(G))

    gdf_new = gpd.GeoDataFrame({"filename": pd.Series([])}, geometry=GeoSeries())

    to_remove = set()
    for connected_indexes_poly in connected_components:
        new_poly = unary_union(gdf.loc[list(connected_indexes_poly), "geometry"])
        filenames = gdf.loc[list(connected_indexes_poly), "filename"].tolist()
        new_data = gpd.GeoDataFrame({"filename": filenames, "geometry": new_poly})
        if not isinstance(new_poly, MultiPolygon):
            gdf_new = pd.concat([gdf_new, new_data], ignore_index=True)
            to_remove.update(connected_indexes_poly)

    gdf = gdf.drop(index=list(to_remove))
    if "filename" in gdf.columns:
        gdf["filename"] = gdf["filename"].apply(lambda x: [x])
    if "filename" in gdf.columns:
        gdf_new["filename"] = gdf_new["filename"].apply(lambda x: [x])

    gdf = pd.concat([gdf, gdf_new], ignore_index=True)
    gdf = gdf[~gdf["geometry"].is_empty].reset_index(drop=True)

    # Buffer the geometry back to original size
    gdf["geometry"] = gdf["geometry"].buffer(-buffer_distance).buffer(buffer_distance)

    return gdf
