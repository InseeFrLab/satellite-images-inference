import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon, Polygon
from geopandas import GeoSeries
import pandas as pd
import networkx as nx
from tqdm import tqdm

predictions = gpd.read_parquet("../data/predictions.parquet")
merged_pred = gpd.read_parquet("../data/merged_pred.parquet")


def check_line_intersection(poly1, poly2):
    intersection = poly1.intersection(poly2)
    if isinstance(intersection, (LineString, MultiLineString)):
        return True
    return False


def touches_on_lines(gdf_original):
    gdf = gdf_original.copy()

    buffer_distance = 3
    gdf["geometry"] = gdf["geometry"].buffer(-buffer_distance).buffer(buffer_distance)
    sindex = gdf.sindex

    touching_pairs = []
    for idx, poly in tqdm(gdf.iterrows(), total=gdf.shape[0], desc="Processing geometries"):
        neighbors_indices = list(sindex.query(poly.geometry, predicate="touches"))
        for neighbor_idx in neighbors_indices:
            if neighbor_idx != idx:
                neighbor_poly = gdf.iloc[neighbor_idx].geometry
                if check_line_intersection(poly.geometry, neighbor_poly):
                    touching_pairs.append((idx, neighbor_idx))

    touching_pairs = list(set(frozenset(pair) for pair in touching_pairs))

    # graphe connecte
    G = nx.Graph()

    for idx1, idx2 in touching_pairs:
        G.add_edge(idx1, idx2)

    connected_components = list(nx.connected_components(G))
    connected_components = [list(elt) for elt in connected_components]

    gdf_new = gpd.GeoDataFrame({"filename": pd.Series([])}, geometry=GeoSeries())

    to_remove = []
    for connected_indexes_poly in connected_components:
        new_poly = Polygon()
        filenames = []
        for idx_poly in connected_indexes_poly:
            new_poly = unary_union([new_poly, gdf.at[idx_poly, "geometry"]])
            filenames.append(gdf.at[idx_poly, "filename"])

        if not isinstance(new_poly, MultiPolygon):
            new_data = gpd.GeoDataFrame({"filename": filenames, "geometry": new_poly})
            gdf_new = pd.concat([gdf_new, new_data], ignore_index=True)
            to_remove += connected_indexes_poly

    gdf = gdf.drop(index=list(set(to_remove)))
    gdf["filename"] = gdf["filename"].apply(lambda x: [x])
    gdf = pd.concat([gdf, gdf_new], ignore_index=True)
    gdf = gdf[~gdf["geometry"].is_empty]
    gdf.reset_index(drop=True, inplace=True)

    buffer_distance = 3
    gdf["geometry"] = gdf["geometry"].buffer(-buffer_distance).buffer(buffer_distance)

    return gdf
