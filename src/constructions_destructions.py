import geopandas as gpd
import pandas as pd
from geopandas import overlay
import requests
import argparse
import time
import math
import numpy as np
from lxml import etree
import io

from src.make_predictions_from_api import save_geopackage_to_s3
from app.utils import get_file_system


def filtre_compacite(table, seuil_compacite=0.08):
    table["compacite"] = (4 * math.pi * table.area) / (table.length**2)
    table_filtree = table[table["compacite"] > seuil_compacite]
    return table_filtree


def filtre_taille(table, seuil_taille=5):
    table = table.copy()
    table["aire_bati"] = table.area
    if seuil_taille != 0:
        table_triee = table.sort_values(by="aire_bati")
        decile_seuil = np.percentile(table_triee["aire_bati"], seuil_taille)
        polygone_decile = table_triee[table_triee["aire_bati"] <= decile_seuil].iloc[-1]
        table_filtree = table[table["aire_bati"] > polygone_decile["aire_bati"]]
    else:
        table_filtree = table
    return table_filtree


def get_build_evol(
    dep: str, year: int, model_name: str, model_version: str, fs
) -> pd.DataFrame:
    global_path = f"projet-slums-detection/data-prediction/PLEIADES/{dep}/"
    paths = fs.ls(global_path, detail=False)

    available_years = []
    pattern = f"/{model_name}/{model_version}/predictions.parquet"

    for path in paths:
        if fs.exists(path+pattern):
            year_available = path.split("/")[-1]
            available_years.append(int(year_available))

    path_end = f"{global_path}{str(year)}{pattern}"
    data_end = gpd.read_parquet(path_end, filesystem=fs)
    data_end = data_end[data_end['label'] == 1][['geometry']].copy()
    data_end["year"] = year
    data_end = data_end.set_geometry('geometry')
    data_end["geometry"] = data_end.geometry.buffer(2.5).buffer(-2.5)

    # Calcul des constructions et destructions
    years_before = [y for y in available_years if y < year]

    if years_before:
        constructions_list = []
        destructions_list = []
        for year_start in years_before:

            path_start = f"{global_path}{str(year_start)}{pattern}"
            data_start = gpd.read_parquet(path_start, filesystem=fs)
            data_start = data_start[data_start['label'] == 1][['geometry']].copy()
            data_start["year"] = year_start
            data_start = data_start.set_geometry('geometry')
            data_start["geometry"] = data_start.geometry.buffer(2.5).buffer(-2.5)

            data_start = data_start[data_start.is_valid]
            data_end = data_end[data_end.is_valid]

            # todo : améliorer la méthode pour filtrer les constructions/destructions
            constructions = overlay(data_end, data_start, how='difference')
            destructions = overlay(data_start, data_end, how='difference')

            constructions = filtre_compacite(constructions)
            destructions = filtre_compacite(destructions)

            constructions = filtre_taille(constructions)
            destructions = filtre_taille(destructions)

            # Constructions
            constructions['year_start'] = year_start
            constructions['evolution'] = 'construction'
            constructions_list.append(constructions[['geometry', 'year_start', 'evolution']])

            # Destructions
            destructions['year_start'] = year_start
            destructions['evolution'] = 'destruction'
            destructions_list.append(destructions[['geometry', 'year_start', 'evolution']])

        # Assemblage final
        evolutions_bati_df = pd.concat(constructions_list+destructions_list, ignore_index=True)

        return_path = f"""{global_path}{str(year)}/{model_name}/{model_version}/"""

        evolutions_bati_df.to_parquet(f"{return_path}evolutions_bati.parquet", filesystem=fs)
        save_geopackage_to_s3(
            evolutions_bati_df,
            f"{return_path}evolutions_bati.gpkg",
            filesystem=fs,
        )
        generate_sld(years_before, fs, f"{return_path}evolution_style.sld")


def ogc_element(ns, tag):
    return etree.Element(f"{{{ns['ogc']}}}{tag}")


def create_rule(ns, year, evolution, color):
    rule = etree.Element("Rule")

    name = etree.Element("Name")
    name.text = f"{year}_{evolution}"
    rule.append(name)

    filter_elem = ogc_element(ns, "Filter")
    and_elem = ogc_element(ns, "And")

    year_eq = ogc_element(ns, "PropertyIsEqualTo")
    year_prop = ogc_element(ns, "PropertyName")
    year_prop.text = "year_start"
    year_val = ogc_element(ns, "Literal")
    year_val.text = str(year)
    year_eq.extend([year_prop, year_val])

    evo_eq = ogc_element(ns, "PropertyIsEqualTo")
    evo_prop = ogc_element(ns, "PropertyName")
    evo_prop.text = "evolution"
    evo_val = ogc_element(ns, "Literal")
    evo_val.text = evolution
    evo_eq.extend([evo_prop, evo_val])

    and_elem.extend([year_eq, evo_eq])
    filter_elem.append(and_elem)
    rule.append(filter_elem)

    polygon_sym = etree.Element("PolygonSymbolizer")
    fill = etree.Element("Fill")
    css_color = etree.Element("CssParameter", name="fill")
    css_color.text = color
    css_opacity = etree.Element("CssParameter", name="fill-opacity")
    css_opacity.text = "0.5"
    fill.extend([css_color, css_opacity])
    polygon_sym.append(fill)
    rule.append(polygon_sym)

    return rule


def generate_sld(years, fs, output_path):
    ns = {
        "ogc": "http://www.opengis.net/ogc",
        "xlink": "http://www.w3.org/1999/xlink",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }

    sld = etree.Element(
        "StyledLayerDescriptor",
        nsmap={None: "http://www.opengis.net/sld", **ns},
        version="1.0.0",
        attrib={
            "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation":
            "http://www.opengis.net/sld StyledLayerDescriptor.xsd"
        }
    )

    named_layer = etree.SubElement(sld, "NamedLayer")
    layer_name = etree.SubElement(named_layer, "Name")
    layer_name.text = "evolution_by_year"

    user_style = etree.SubElement(named_layer, "UserStyle")
    title = etree.SubElement(user_style, "Title")
    title.text = "Évolution par année (construction/destruction)"

    feature_type_style = etree.SubElement(user_style, "FeatureTypeStyle")

    # Règles par année
    for year in years:
        feature_type_style.append(create_rule(ns, year, "construction", "#00cc44"))
        feature_type_style.append(create_rule(ns, year, "destruction", "#cc0033"))

    tree = etree.ElementTree(sld)
    buffer = io.BytesIO()
    tree.write(buffer, encoding="UTF-8", xml_declaration=True, pretty_print=True)
    buffer.seek(0)

    # Écriture sur S3
    with fs.open(output_path, 'wb') as f:
        f.write(buffer.read())

    print(f"✅ SLD généré avec succès : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make constructions/destructions")
    parser.add_argument("--dep", type=str, required=True, help="Département (e.g., 'MAYOTTE')")
    parser.add_argument("--year", type=int, required=True, help="Année (e.g., 2017)")
    args = parser.parse_args()

    dep = args.dep
    year = args.year

    start_time = time.time()

    # Get info of the model
    model_info = requests.get("https://satellite-images-inference.lab.sspcloud.fr/").json()

    fs = get_file_system()

    get_build_evol(dep, year, model_info["model_name"], model_info["model_version"], fs)

    end_time = time.time() - start_time
    print(f"{dep} constructions/destructions in {round(end_time/60)} min")
